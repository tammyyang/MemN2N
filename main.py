import sys
import logging
import memn2n


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


def print_help():
    parms = {'-e/--explain': 'print the usage of argument json file',
             '-h/--help': 'print help messages'}
    output = ''
    for k in parms.keys():
        output += '  [' + k + ']: ' + parms[k] + '\n'
    print('python main.py\n%s' % output)


def main():
    explain = None
    args = sys.argv[1:]
    if len(args) > 0 and args[0] in ['-h', '--help']:
        print_help()
        sys.exit(1)

    model = memn2n.Model()
    if len(args) > 0:
        if args[0] in ['-e', '--explain']:
            if len(args) > 1:
                explain = args[1]
            else:
                explain = ''
            model.args.explain_args('rnn.example.json', arg=explain)
            sys.exit(1)
    model.train()

if __name__ == '__main__':
    main()
