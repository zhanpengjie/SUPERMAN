import FaceR
import json
import math
import numpy as np
import os
import pickle
from sklearn import svm
import sys
import tensorflow as tf


# A whitelist for valid file format
whitelist = ('.png', '.jpg', '.jpeg')
image_list = ('linktime.png', 'linktime.jpg', 'linktime.jpeg')

# Descending quick_sort
def quick_sort(list, left , right):
    if left >= right:
        return
    low = left
    high = right
    key = list[low]
    while left < right:
        while left < right and list[right] <= key:
            right = right - 1
        list[left] = list[right]
        while left < right and list[left] >= key:
            left = left + 1
        list[right] = list[left]
    list[right] = key
    quick_sort(list, low, left-1)
    quick_sort(list, right+1, high)


def process(show_data_dir, aligned_data_dir, model, classifier_filename, mode, batch_size=90, image_size=160):

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # np.random.seed(seed=args.seed)

            dataset = FaceR.get_dataset(aligned_data_dir)

            paths, labels = FaceR.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset), file=sys.stderr)
            print('Number of images: %d' % len(paths), file=sys.stderr)

            # Load the model
            print('Loading feature extraction model', file=sys.stderr)
            FaceR.load_model(model)
            # FaceR.load_model('/Users/lzh/models/facenet/20180408-102900/20180408-102900.pb')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images', file=sys.stderr)
            nrof_images = len(paths)
            nrof_batches_per_ecoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_ecoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = FaceR.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # Train the classifer
            if (mode == 'TRAIN'):

                # Use SVM as the algorithm of the classifier
                clf = svm.SVC(kernel='linear', probability=True)

                # Fit the features and labels to SVM
                clf.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((clf, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp, file=sys.stderr)

            # Classify images
            elif (mode == 'CLASSIFY'):

                print('Classify images', file=sys.stderr)

                # Load the trained classifier
                with open(classifier_filename_exp, 'rb') as infile:
                    (clf, class_names) = pickle.load(infile)

                print('Loaded classifier from file "%s"' % classifier_filename_exp, file=sys.stderr)

                # Predict the new image by using its face features
                predictions = clf.predict_proba(emb_array)

                probs = []
                for i in range(len(predictions)):
                    probs.append(list(zip(predictions[0], clf.classes_)))
                probs = probs[0]

                # Pick out the top 4 probabilities with their label
                quick_sort(probs, 0, len(probs) - 1)

                # Use tmp to store the dic of the result
                tmp = []
                for i in range(4):
                    pro = round(probs[i][0] * 100, 1)

                    # Get the data directory containing target image
                    target_data_dir = os.path.join(show_data_dir, class_names[probs[i][1]].replace(' ', '_'))
                    # Get the target image path
                    images = os.listdir(target_data_dir)

                    local_paths = [os.path.join(target_data_dir, img) for img in images \
                                   if img in image_list]

                    dict = {'Name': class_names[probs[i][1]], 'probability': pro, 'Path': local_paths[0]}
                    tmp.append(dict)

                res = json.dumps(tmp, ensure_ascii=False)
                print(res, end='')


if __name__ == '__main__':
    pass