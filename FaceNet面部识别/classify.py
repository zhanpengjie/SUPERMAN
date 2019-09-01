import alignment
import argparse
import classifier
import os
import shutil


if __name__ == '__main__':
    # Parameters for classify
    parser = argparse.ArgumentParser()

    parser.add_argument('new_data_dir', type=str,
                        help="Path to the new image uploaded by user.")
    parser.add_argument('test_data_dir', type=str,
                        help="Path to the data directory containing the aligned new images.")
    parser.add_argument('raw_data_dir', type=str,
                        help="Path to the data directory containing the face images which will be shown.")
    parser.add_argument('user_id', type=str,
                        help="The id of the user.")
    parser.add_argument('model', type=str,
                        help="Path to the data directory containing the pre-trained model.")
    parser.add_argument('classifier_filename', type=str,
                        help="Path to the data directory containing the classifier.")
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)

    args = parser.parse_args()

    aligned_data_dir = os.path.join(args.test_data_dir, args.user_id)
    if os.path.exists(aligned_data_dir):
        shutil.rmtree(aligned_data_dir)
    os.mkdir(aligned_data_dir)

    # Align the new image before classifying
    alignment.process(args.new_data_dir, aligned_data_dir)

    # Classify the aligned new image
    classifier.process(args.raw_data_dir, aligned_data_dir, args.model, args.classifier_filename, 'CLASSIFY')

    # Delete the image in test data dir after classifying
    shutil.rmtree(aligned_data_dir)