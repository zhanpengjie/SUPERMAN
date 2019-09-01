import argparse
import classifier
import alignment

if __name__ == '__main__':
    # Parameters for training
    parser = argparse.ArgumentParser()

    parser.add_argument('raw_data_dir', type=str,
                        help="Path to the data directory containing the aligned face images.")
    parser.add_argument('aligned_data_dir', type=str,
                        help="Path to the data directory containing the aligned face images.")
    parser.add_argument('model', type=str,
                        help="Path to the data directory containing the pre-trained model.")
    parser.add_argument('classifier_filename', type=str,
                        help="Path to the data directory containing the classifier.")

    args = parser.parse_args()

    # Align the raw images before training
    alignment.process(args.raw_data_dir, args.aligned_data_dir)

    # Train the aligned images
    classifier.process(args.raw_data_dir, args.aligned_data_dir, args.model, args.classifier_filename, 'TRAIN')
