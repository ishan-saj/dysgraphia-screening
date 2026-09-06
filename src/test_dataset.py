from dataset import DysgraphiaDataset


dataset = DysgraphiaDataset(
    metadata_csv="../data/metadata/metadata_all.csv",
    image_root="../data/raw/IAM/images"
)

print("\nNumber of samples:")
print(len(dataset))

sample = dataset[0]

print("\nSample information:")
print("Sample ID:", sample["sample_id"])
print("Image path:", sample["image_path"])

print("\nTensor shapes:")
print("Image:", sample["image"].shape)
print("OpenCV features:", sample["opencv_features"].shape)
print("Ink mask:", sample["ink_mask"].shape)

print("\nOpenCV features:")
print(sample["opencv_features"])