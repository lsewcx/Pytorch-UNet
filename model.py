import sys
sys.path.append('../')
from pycore.tikzeng import *

# 定义你的网络结构
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_Conv('depthwise', s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption="depthwise")
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()