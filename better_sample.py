"""
5/22/23
better_sample.py
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import random
from data_generation import generate_problems

import time
import numpy as np
import statistics
import math
from scipy import stats
import string

# -----------------------------------------------------------------------------
def convert_base(num, old_symbols, new_symbols):
    """
    Convert a number from one base to another
    :param num: string representation of the number
    :param old_symbols: list of characters representing the old base
    :param new_symbols: list of characters representing the new base
    :return: string representation of the number in the new base
    """
    old_base = len(old_symbols)
    new_base = len(new_symbols)
    num_in_base_10 = 0
    for i, digit in enumerate(reversed(num)):
        num_in_base_10 += old_symbols.index(digit) * (old_base ** i)
    result = ''
    while num_in_base_10 > 0:
        result = new_symbols[num_in_base_10 % new_base] + result
        num_in_base_10 //= new_base
    return result if result else new_symbols[0]

# args
# num_samples and out_dir
# num_samples is number of tests to run on the model. higher means more precise results
# out_dir is the directory of the model to test
def sample_model(config, operators):
    # old args: base_symbols=None, num_samples=10, out_dir='out-add-math-char', operators=['+']
    """
    Sample from a trained model
    :param config: dictionary of configuration parameters including:
    base_symbols: The symbols to use for the base
    num_samples: The number of samples to generate from the model
    out_dir: The directory of the model to test
    :param operators: The operators to use in the problems
    :return results: A dictionary of the results of the tests
    """
    # -----------------------------------------------------------------------------
    out_dir = config['out_dir']
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        print("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join(out_dir, 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        print(f"the vocab is {meta['stoi']}")
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)


    # max_new_tokens = 128  # block size
    max_new_tokens = config['block_size']
    num_samples = config['num_samples']  # number of times to sample from the model
    temperature = config['temperature']  # temperature of the sampling
    test_problems = []  # list to store all the test problems
    test_answers = []  # list to store all the test answers
    model_answers = []  # list to store all the model answers
    base_symbols = config['base_symbols']
    base = config['base']
    # operators = config['operators']
    # CURRENT BASE LIMITATIONS: can go up to 26, down to 2, uses lowercase letters for symbols

    # Loop through num_samples
    verbose = True
    start = time.time()
    print(f"Generating {num_samples} samples...")
    for i in range(num_samples):
        # print(f"### Generating response {i + 1}/{num_samples} ###")

        # create a prompt using some problems
        prompt = generate_problems(max_new_tokens, num_samples=1, base_symbols=base_symbols, problem_operators=operators)
        # convert list to string
        prompt = ''.join(prompt)
        # print(f"*** original prompt {i+1}:\n{prompt}")

        # FOR TESTING: use a specific prompt
        # print(f"current prompt: \n\'{prompt}\'")
#         prompt = """
# az + zz = cz"""
#
#         print(f"current prompt: \n\'{prompt}\'")
        """The current prompt is of the form:
        cb + vn = xo
        uovlwb + xz = uovmua"""
        # save the answer to the last problem (in the example, this would be "uovmua")
        last_problem = prompt[prompt.rfind('\n'):]
        test_problems.append(last_problem)
        # print(f"last_problem: \'{last_problem}\'")
        last_answer = last_problem[last_problem.rfind('=') + 2:]
        test_answers.append(last_answer)
        # print(f"last_answer: \'{last_answer}\'")
        # now remove the last answer from the prompt (so now the last line is just "uovlwb + xz = ")
        prompt = prompt[:prompt.rfind(last_answer)]
        # print(f"*** new prompt {i+1}:\n{prompt}")
        # record the index of the start of the answer and the end of the answer
        # this is used to extract the completed last problem from the response
        prompt_end_index = len(prompt)
        completed_problem_end = prompt_end_index + len(last_answer)
        # print(f"start: {prompt_end_index}, end: {completed_problem_end}")

        # Encode prompt
        start_ids = encode(prompt)

        # Create input tensor
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        # Generate response
        new_tokens = max_new_tokens - len(start_ids)
        # print(f"generating {new_tokens} tokens...")
        with torch.no_grad():
            with ctx:
                y = model.generate(x, new_tokens, temperature=temperature, top_k=top_k)

        # Decode response
        response = decode(y[0].tolist())

        # now extract the answer to the last problem
        # if the model's answer is longer than the correct answer, then we need to include the entire answer
        # otherwise, just include up to completed_problem_end
        if len(response) > completed_problem_end:
            # the model's answer should only be 1 character longer than the correct answer
            model_answer = response[prompt_end_index:completed_problem_end + 1]
        else:
            model_answer = response[prompt_end_index:completed_problem_end]
        # split on the newline
        model_answer = model_answer.split('\n')[0]
        # remove any whitespace
        model_answer = model_answer.strip()
        # for the first 5 responses, print the response and the answer
        if i < 5 and verbose:
            print(f"response {i + 1}: \'{response}\'")
            # print(f"model_answer: \'{model_answer}\'")
        model_answers.append(model_answer)

    # end timer
    end = time.time()
    t_total = end - start
    num_per_sec = round(num_samples / t_total, 2)
    print(f"Sampling time: {round(t_total, 3)} seconds ({num_per_sec} samples per second)")



    # ----------------------------
    # PROBLEM VERIFICATION
    # ----------------------------
    # print("\n### Problem Verification ###\n")

    prob_num = 0
    num_correct = 0
    num_skipped = 0
    distance_list = []  # list to store the edit distances
    off_by_one_list = []  # list to store the number of off-by-one errors
    operator = test_problems[0].split(' ')[1]
    # print an error if this is not '+'
    if operator != '+':
        print(f"ERROR: operator is {operator}, but this code only works for addition problems")
        operator = '+'
    for answer in model_answers:
        prob_num += 1
        # first make sure the answer does not contain any operators, equals signs, spaces, or newlines
        if operator in answer or '=' in answer or ' ' in answer or '\n' in answer:
            num_skipped += 1
            # for the first 5 problems, print the error and the answer
            if prob_num <= 5 and verbose:
                print(f"skipping answer \'{answer}\' because it contains an operator, equals sign, space, or newline")
                # try and find what is wrong with the answer
                if operator in answer:
                    print(f"answer contains an operator")
                if '=' in answer:
                    print(f"answer contains an equals sign")
                if ' ' in answer:
                    print(f"answer contains a space")
                if '\n' in answer:
                    print(f"answer contains a newline")
            continue
        # both the answer and the test answer should be the same length and in the same base
        # but we can just skip the ones that are more than 1 character longer than the test answer
        if len(answer) > (len(test_answers[prob_num - 1]) + 1):
            num_skipped += 1
            print(f"skipping answer {answer} because it is at least 2 characters longer than the test answer {test_answers[prob_num - 1]}")
            continue

        # does the answer equal the test answer
        is_correct_answer = (answer == test_answers[prob_num - 1])
        # for the first 5 problems, print the problem and the answer
        if prob_num <= 5 and verbose:
            print(f"Problem {prob_num}: {test_problems[prob_num - 1]} = {test_answers[prob_num - 1]}")  # verify the problem
            print(f"Model answer: {answer}")

        # function to compare two strings - naive implementation
        # returns the number of changes needed to make the strings the same
        # example: compare_strings("abc", "abd") returns 1 and compare_strings("abc", "add") returns 3
        def edit_distance(str1, str2):
            distance = 0
            shortest_len = min(len(str1), len(str2))
            for i in range(shortest_len):
                distance += abs(ord(str1[i]) - ord(str2[i]))
            return distance

        # determine if the difference is a power of the base (base^0, base^1, base^2, etc.)
        def is_difference_base_pow(str1, str2, base_symbols):
            # convert both strings to base 10
            base10_symbols = '0123456789'
            num1 = convert_base(str1, base_symbols, base10_symbols)
            num1 = int(num1)
            num2 = convert_base(str2, base_symbols, base10_symbols)
            num2 = int(num2)
            # calculate the distance between the two numbers
            distance = abs(num1 - num2)
            base = len(base_symbols)
            # calculate the largest power of the base that is less than the distance
            power = 0
            while base ** power < distance:
                power += 1
            # if the distance is a power of the base, then the answer is correct
            return base ** power == distance

        # calculate the distance between the model answer and the test answer
        distance = edit_distance(answer, test_answers[prob_num - 1])
        distance_list.append(distance)
        off_by_one_list.append(is_difference_base_pow(answer, test_answers[prob_num - 1], base_symbols))
        # print(f"distance: {distance}")

        if is_correct_answer:
            num_correct += 1
            # print(f"Problem {prob_num} is correct!")
        else:
            # print(f"Problem {prob_num} is incorrect!")
            # print(f"Model answer: {answer}")
            # print(f"Test answer: {test_answers[prob_num - 1]}")
            pass

    # print results
    accuracy = round(num_correct / prob_num * 100, 2)
    # print(f"number correct: {num_correct} out of {prob_num}")
    # print(f"accuracy: {accuracy}%")
    # print(f"number skipped: {num_skipped} out of {prob_num}")
    if len(distance_list) < 2:
        distance_list = [0, 0, 0]
    med_distance = statistics.median(distance_list)
    mean_distance = statistics.mean(distance_list)
    var_distance = statistics.variance(distance_list)
    # print(f"median letters distance: {med_distance}")
    # print(f"mean distance: {mean_distance}")
    # print(f"variance distance: {var_distance}")

    # fraction skipped
    num_incorr = prob_num - num_correct
    if num_incorr == 0:
        fraction_skipped = 0
    else:
        fraction_skipped = round(num_skipped / num_incorr * 100, 2)
    # fraction off by one
    num_off_by_one = sum(off_by_one_list)
    if num_incorr == 0:
        fraction_off_by_one = 0
    else:
        fraction_off_by_one = round(num_off_by_one / num_incorr * 100, 2)

    # store the accuracy, fraction skipped, median distance, mean distance, and variance distance in a dictionary
    results = {
        'accuracy': accuracy,
        'frac_skipped': fraction_skipped,
        'frac_off_by_one': fraction_off_by_one,
        'median_distance': med_distance,
        'mean_distance': mean_distance,
        'variance_distance': var_distance
    }
    return results



# if this file is run directly
if __name__ == '__main__':
    print("## Running better_sample.py directly ##")
    # run def sample_model(base_symbols=string.ascii_lowercase, num_samples=10, out_dir='out-add-math-char')
    out_dir = 'test-out-add-math-char-0.0003-lr-26-base'
    samples = 1
    # symbols = string.ascii_lowercase
    base_symbols = string.ascii_lowercase
    base = len(base_symbols)
    base_symbols = base_symbols[-1:] + base_symbols[:-1]  # right shift
    print(f'base: {base} symbols: {base_symbols}')
    operators = ['+']
    results = sample_model(base_symbols=base_symbols, num_samples=samples, out_dir=out_dir, operators=operators)
    print(results)


