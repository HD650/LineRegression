from PIL import Image
import pickle
import numpy as np


def open_pickled_data(filename):
    print("reading {0}...".format(filename))
    f = open(filename, 'rb')
    image_data = pickle.load(f, encoding='latin1')
    f.close()
    print("finished")
    print("set size: {0}".format(len(image_data["labels"])))
    return image_data


def reorganize_image(image_data):
    label_bufferr = np.zeros((len(image_data["data"]), 10), np.int8)
    for index, item in enumerate(image_data['data']):
        red = item[0:32*32]
        green = item[32*32:2*32*32]
        blue = item[32*32*2:32*32*3]
        pixel_buffer = np.ndarray(shape=3072, dtype=np.uint8)
        for i in range(32*32-1):
            pixel_buffer[i*3] = red[32*32-1-i]
            pixel_buffer[i*3+1] = green[32*32-1-i]
            pixel_buffer[i*3+2] = blue[32*32-1-i]
        image_data['data'][index] = pixel_buffer
        print("sample{1} {0} pixel arrangement finished".format(image_data["filenames"][index], str(index)))
        label_bufferr[index, image_data["labels"][index]] = 1
        print("sample{1} {0} pixel label finished".format(image_data["filenames"][index], str(index)))
    image_data["labels"] = label_bufferr
    return image_data


def save_as_jpg(image_data, count):
        for i in range(len(image_data['data'])):
            temp_image = Image.frombuffer("RGB", (32, 32), image_data['data'][i])
            temp_image.save("./image/"+image_data["filenames"][i]+".jpg")
            print("saving "+image_data["filenames"][i])
            if i > count:
                print("done")
                return


def save_as_batch(image_data, file_name):
    print("opening file...")
    f = open(file_name, 'wb')
    print("writing pickled data...")
    f.write(pickle.dumps(image_data))
    f.close()
    print("finished")


if __name__ == '__main__':
    raw_data = open_pickled_data("./data/data_batch_1")
    reorganized_data = reorganize_image(raw_data)
    save_as_batch(reorganized_data, "./data/reorganized_batch1")

    raw_data = open_pickled_data("./data/data_batch_2")
    reorganized_data = reorganize_image(raw_data)
    save_as_batch(reorganized_data, "./data/reorganized_batch2")

    raw_data = open_pickled_data("./data/data_batch_3")
    reorganized_data = reorganize_image(raw_data)
    save_as_batch(reorganized_data, "./data/reorganized_batch3")

    raw_data = open_pickled_data("./data/data_batch_4")
    reorganized_data = reorganize_image(raw_data)
    save_as_batch(reorganized_data, "./data/reorganized_batch4")

    raw_data = open_pickled_data("./data/data_batch_5")
    reorganized_data = reorganize_image(raw_data)
    save_as_batch(reorganized_data, "./data/reorganized_batch5")

    raw_data = open_pickled_data("./data/test_batch")
    reorganized_data = reorganize_image(raw_data)
    save_as_batch(reorganized_data, "./data/reorganized_test_batch")

