import argparse

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--arg1')
   parser.add_argument('--arg2')
   args = parser.parse_args()

   print(args.arg1)
   print(args.arg2)

   my_dict = {'arg1': args.arg1, 'arg2': args.arg2}
   print(my_dict)
