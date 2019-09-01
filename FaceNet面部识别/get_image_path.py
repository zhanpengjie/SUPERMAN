import argparse
import os
import shutil
import sys


def process(args):

    user_dir = args.user_id + '-' + args.tag.replace(' ', '_')
    image_path = os.path.join(args.raw_data_dir, user_dir)
    #show_path = os.path.join(args.show_data_dir, user_dir)

    if os.path.exists(image_path):
        if os.path.isfile(image_path):
            print('Error!!! The path should be a path of a directory.', file=sys.stderr)
    else:
        os.makedirs(image_path)

    new_image = os.path.join(args.new_data_dir,str(os.listdir(args.new_data_dir)[-1]), args.image_name)

    shutil.copy(new_image, image_path)
    #shutil.copy(new_image, show_path)

    if os.path.exists(os.path.join(image_path, args.image_name)):
        os.remove(new_image)
        return 0
    else:
        return -1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('raw_data_dir', type=str,
                        help="Path to the data directory containing the aligned images.")
    parser.add_argument('new_data_dir', type=str,
                        help="Path to the data directory containing the aligned images.")
    parser.add_argument('user_id', type=str,
                        help="The id of the user.")
    parser.add_argument('tag', type=str,
                        help="Tag of the new image.")
    parser.add_argument('image_name', type=str,
                        help="Name of the new image.")

    return parser.parse_args(argv)


# Get new images and its path, and move it to the training dataset
if __name__ == '__main__':
    process(parse_arguments(sys.argv[1:]))