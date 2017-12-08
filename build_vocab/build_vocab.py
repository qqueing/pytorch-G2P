"""Standalone script to generate word vocabularies from monolingual corpus."""

import argparse

import data_utils


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--model_dir", default="../prepared_data/", help="Data for training")
  parser.add_argument(
      "--train_file", default="../naive_data/train", help="Data for training")
  parser.add_argument(
      "--valid_file", default="", help="Data for validation")
  parser.add_argument(
      "--test_file", default="", help="Data for test")

  args = parser.parse_args()

  data_utils.prepare_g2p_from_naive_data(args.model_dir,args.train_file, args.valid_file,args.test_file)

if __name__ == "__main__":
  main()
