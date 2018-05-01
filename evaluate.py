# coding=utf8
import os
import numpy

# ref: https://github.com/chsasank/ATIS.keras/blob/master/metrics/accuracy.py
def conlleval(p, g, w, filename):
    
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    with open(filename, 'w') as f:
        f.writelines(out)
    return get_metrics(filename)

def get_metrics(filename):
    if os.path.exists(filename):
        cmd = './conlleval.pl < %s | grep accuracy' % filename
        out = os.popen(cmd).read().split()
        os.system('rm %s' % filename)  # delete file
        return {'Precision': float(out[3][:-2]), 'Recall': float(out[5][:-2]), 'F1': float(out[7])}

    return None
