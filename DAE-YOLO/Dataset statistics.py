import os

dataset_root =  ''

pixel_size_bins = [0, 50, 100, 150, 200,1024]

object_count_bins = [0,10, 100, 200, 300,700]

pixel_size_distribution = {i: 0 for i in range(len(pixel_size_bins) - 1)}
object_count_distribution = {i: 0 for i in range(len(object_count_bins) - 1)}
image_object_count = {}

for part in [ 'train', 'val']:
    labels_dir = os.path.join(dataset_root, 'labels', part)


    for label_name in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_name)
        image_name = os.path.splitext(label_name)[0]
        object_count = 0

        with open(label_path, 'r') as f:
            lines = f.readlines()
            object_count = len(lines)

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:

                    _, _, _, width, height = map(float, parts)
                    image_width, image_height = 1024, 1024
                    target_pixel_width = width * image_width
                    target_pixel_height = height * image_height
                    target_pixel_size = max(target_pixel_width , target_pixel_height)

                    for i in range(len(pixel_size_bins) - 1):
                        if pixel_size_bins[i] <= target_pixel_size < pixel_size_bins[i + 1]:
                            pixel_size_distribution[i] += 1
                            break

        image_object_count[image_name] = object_count

        for i in range(len(object_count_bins) - 1):
            if object_count_bins[i] <= object_count < object_count_bins[i + 1]:
                object_count_distribution[i] += 1
                break

print("Target pixel size distribution:")
for i in range(len(pixel_size_bins) - 1):
    bin_range = (pixel_size_bins[i], pixel_size_bins[i + 1])
    print(f"interval: {bin_range}, number: {pixel_size_distribution[i]}")

print("\nInterval distribution of image target quantity:")
for i in range(len(object_count_bins) - 1):
    bin_range = (object_count_bins[i], object_count_bins[i + 1])
    print(f"Interval: {bin_range}, Number of pictures:{object_count_distribution[i]}")