#!/usr/bin/python

#
# distort.py
#
# License: Public Domain (www.unlicense.org)
#
# This is free and unencumbered software released into the public domain.
# Anyone is free to copy, modify, publish, use, compile, sell, or distribute this 
# software, either in source code form or as a compiled binary, for any purpose, 
# commercial or non-commercial, and by any means.
# In jurisdictions that recognize copyright laws, the author or authors of this 
# software dedicate any and all copyright interest in the software to the public 
# domain. We make this dedication for the benefit of the public at large and to 
# the detriment of our heirs and successors. We intend this dedication to be an 
# overt act of relinquishment in perpetuity of all present and future rights to 
# this software under copyright law.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import sys
import numpy as np
import scipy as sp
import scipy.io.wavfile as wav


def convert_to_float32_array(data):
    info = np.iinfo(data.dtype)
    data = data.astype('float32', copy=False)
    data[np.sign(data)==-1] *= (1.0 / abs(info.min))
    data[np.sign(data)== 1] *= (1.0 / abs(info.max))
    return data


def convert_from_float32_array(data, dtype):
    info = np.iinfo(dtype)
    data[np.sign(data)==-1] *= abs(info.min)
    data[np.sign(data)== 1] *= abs(info.max)
    data = data.astype('int16', copy=False)
    return data


def get_intervals(array, min_length=0):
    indexes = []

    prev = 0.0
    last_i = 0
    for i, curr in enumerate(np.nditer(array)):
        if prev <= 0.0 and curr > 0.0 and (i - last_i) >= min_length:
            indexes.append(i)
            last_i = i
        prev = curr

    intervals = []
    for i in range(1, len(indexes)):
        intervals.append({'start': indexes[i-1], 'end': indexes[i]})
    return intervals


def distort_repeat(src, n):
    '''repeat waveset'''
    intervals = get_intervals(src)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    result = []

    if intervals_start != 0:
        result.append(src[0:intervals_start])

    for interval in intervals:
        result.append(np.tile(src[interval['start']:interval['end']], n))

    if intervals_end != len(src):
        result.append(src[intervals_end:len(src)])

    return np.concatenate(result)


def distort_interpolate(src, n):
    '''repeat waveset by interpolate between each of them'''
    intervals = get_intervals(src, 256)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    result = []

    if intervals_start != 0:
        result.append(src[0:intervals_start])

    L = len(intervals)
    for i in range(1, L+1):
        last = src[intervals[i-1]['start']:intervals[i-1]['end']]
        curr = src[intervals[i%L]['start']:intervals[i%L]['end']]
        diff = len(curr) - len(last)

        result.append(last)
        for j in range(1, n):
            t = float(j) / n
            size = int(round(diff*t)) + len(last)
            sigs = []
            for u in range(0, size):
                pos_curr = int((float(u) / size) * len(curr))
                pos_last = int((float(u) / size) * len(last))
                sig = last[pos_last] * (1.0 - t) + curr[pos_curr] * t
                sigs.append(sig)
            result.append(np.array(sigs, dtype=np.float32))

    if intervals_end != len(src):
        result.append(src[intervals_end:len(src)])

    return np.concatenate(result)


def distort_reverse(src, min_length=0):
    '''reverse waveset'''
    intervals = get_intervals(src, min_length=min_length)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    result = []

    if intervals_start != 0:
        result.append(src[0:intervals_start])

    for interval in intervals:
        s = interval['start']
        e = interval['end']
        src[s:e] = np.fliplr([src[s:e]])[0]
        result.append(src[s:e])

    if intervals_end != len(src):
        result.append(src[intervals_end:len(src)])

    return np.concatenate(result)


def distort_omit(src, n, min_length=0, skip=False):
    '''zeroing/skipping every nth waveset'''
    intervals = get_intervals(src, min_length=min_length)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    result = []

    if intervals_start != 0:
        result.append(src[0:intervals_start])

    i = 0
    for interval in intervals:
        if i == n:
            if not skip:
                result.append(src[interval['start']:interval['end']]*0.0)
            i = 0
        else:
            result.append(src[interval['start']:interval['end']])
            i += 1

    if intervals_end != len(src):
        result.append(src[intervals_end:len(src)])

    return np.concatenate(result)


def distort_delete(src, n, min_length=0):
    return distort_omit(src, n, min_length=min_length, skip=True)


def distort_montage(src, nb_divisions):
    '''shuffle the parts of waveset'''
    import random

    def get_subdivisions(interval, nb_divisions):
        s = interval['start']
        e = interval['end']
        n = e - s
        n_per_division = n / nb_divisions
        result = []
        for i in range(s, e, n_per_division):
            if (i + n_per_division) >= e:
                result.append({'start': i, 'end': e})
            else:
                result.append({'start': i, 'end': i + n_per_division})
        return result

    intervals = get_intervals(src, min_length=512)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    intervals_src = np.copy(src)
    result = []

    if intervals_start != 0:
        result.append(intervals_src[0:intervals_start])

    for interval in intervals:
        subdivisions = get_subdivisions(interval, nb_divisions)
        random.shuffle(subdivisions)
        for d in subdivisions:
            result.append(src[d['start']:d['end']])

    if intervals_end != len(src):
        result.append(intervals_src[intervals_end:len(src)])

    return np.concatenate(result)


def distort_fractal(src, nb_harmonics, gain):
    '''superimpose waveset'''
    intervals = get_intervals(src)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    intervals_src = np.copy(src)
    result = []

    if intervals_start != 0:
        result.append(intervals_src[0:intervals_start])

    for interval in intervals:
        s = interval['start']
        e = interval['end']
        n = e - s
        for i in range(2, nb_harmonics+1):
            intervals_src[s:e] += np.tile(intervals_src[s:e:i], i)[:n]
            intervals_src[s:e] *= gain
        result.append(intervals_src[s:e])

    if intervals_end != len(src):
        result.append(intervals_src[intervals_end:len(src)])

    np.tanh(intervals_src, intervals_src)
    intervals_src *= 0.7

    return np.concatenate(result)


def distort_fractal2(src, nb_harmonics, gain):
    '''superimpose waveset (slightly)'''
    intervals = get_intervals(src)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    intervals_src = np.copy(src)
    result = []

    if intervals_start != 0:
        result.append(intervals_src[0:intervals_start])

    for interval in intervals:
        s = interval['start']
        e = interval['end']
        n = e - s
        for i in range(2, nb_harmonics+1):
            intervals_src[s:e] += np.tile(intervals_src[s:e:i], i)[:n]
            intervals_src[s:e] *= gain
        result.append(intervals_src[s:e])

    if intervals_end != len(src):
        result.append(intervals_src[intervals_end:len(src)])

    np.tanh(src*90.0 + intervals_src + 0.1, intervals_src)
    intervals_src *= 0.15

    return np.concatenate(result)


def distort_cubic(src, gain=10.0, attenuate=0.2):
    '''cubic distortion'''
    intervals = get_intervals(src)
    if len(intervals) == 0:
        return

    intervals_start = intervals[0]['start']
    intervals_end = intervals[-1]['end']
    intervals_src = np.copy(src)
    result = []

    if intervals_start != 0:
        result.append(intervals_src[0:intervals_start])

    def f(x, alpha, beta, gamma, zeta):
        return alpha*x**3.0 + beta*x**2.0 + gamma*x + zeta

    np.tanh(src * gain, intervals_src)
    intervals_src = f(intervals_src, 0.2, 0.5, 0.7,
            np.sin(np.arange(intervals_src.size)*2*np.pi*(22.0/44100))*0.05)

    for interval in intervals:
        s = interval['start']
        e = interval['end']
        n = e - s
        result.append(intervals_src[s:e])

    if intervals_end != len(src):
        result.append(intervals_src[intervals_end:len(src)])

    intervals_src *= attenuate

    return np.concatenate(result)


def distort_repeat_main(argv):
    if len(argv) < 3:
        print('usage: distort repeat in out n')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_n = int(argv[2])

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_repeat(data, arg_n)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_interpolate_main(argv):
    if len(argv) < 3:
        print('usage: distort interpolate in out n')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_n = int(argv[2])

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_interpolate(data, arg_n)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_reverse_main(argv):
    if len(argv) < 2:
        print('usage: distort reverse in out [min_length]')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_min_length = int(argv[2]) if len(argv) > 2 else 0

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_reverse(data, min_length=arg_min_length)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_omit_main(argv):
    if len(argv) < 3:
        print('usage: distort omit in out n [min_length]')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_n = int(argv[2])
    arg_min_length = int(argv[3]) if len(argv) > 3 else 0

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_omit(data, arg_n, min_length=arg_min_length)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_delete_main(argv):
    if len(argv) < 3:
        print('usage: distort delete in out n [min_length]')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_n = int(argv[2])
    arg_min_length = int(argv[3]) if len(argv) > 3 else 0

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_delete(data, arg_n, min_length=arg_min_length)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_montage_main(argv):
    if len(argv) < 3:
        print('usage: distort montage in out nb_divisions')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_nb_divisions = int(argv[2])

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_montage(data, arg_nb_divisions)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_fractal_main(argv):
    if len(argv) < 4:
        print('usage: distort fractal in out nb_harmonics gain')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_nb_harmonics = int(argv[2])
    arg_gain = float(argv[3])

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_fractal(data, arg_nb_harmonics, arg_gain)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


def distort_cubic_main(argv):
    if len(argv) < 2:
        print('usage: distort cubic in out [gain=10.0] [attenuate=0.2]')
        return

    arg_in = argv[0]
    arg_out = argv[1]
    arg_gain = float(argv[2]) if len(argv) > 2 else 10.0
    arg_attenuate = float(argv[3]) if len(argv) > 3 else 0.2

    rate, data = wav.read(arg_in)
    data = convert_to_float32_array(data)
    result = distort_cubic(data, gain=arg_gain, attenuate=arg_attenuate)
    result = convert_from_float32_array(result, np.int16)
    wav.write(arg_out, rate, result)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print('usage: distort <subcmd>')
        sys.exit(1)

    if argv[0] == 'repeat':
        distort_repeat_main(argv[1:])
    elif argv[0] == 'interpolate':
        distort_interpolate_main(argv[1:])
    elif argv[0] == 'reverse':
        distort_reverse_main(argv[1:])
    elif argv[0] == 'omit':
        distort_omit_main(argv[1:])
    elif argv[0] == 'delete':
        distort_delete_main(argv[1:])
    elif argv[0] == 'montage':
        distort_montage_main(argv[1:])
    elif argv[0] == 'fractal':
        distort_fractal_main(argv[1:])
    elif argv[0] == 'cubic':
        distort_cubic_main(argv[1:])
    else:
        print('error: no such subcmd "{}"'.format(argv[0]))
        print('usage: distort <subcmd>')
        sys.exit(1)
