#[cfg(test)]
pub mod simplified;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use candle::{DType, Device, Tensor};
    use parquet::file::reader::SerializedFileReader;

    // NOTE: Waiting on https://github.com/rust-lang/mdBook/pull/1856
    #[rustfmt::skip]
    #[tokio::test]
    async fn book_hub_1() {
// ANCHOR: book_hub_1
use candle::Device;
use hf_hub::api::tokio::Api;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());

let weights_filename = repo.get("model.safetensors").await.unwrap();

let weights = candle::safetensors::load(weights_filename, &Device::Cpu).unwrap();
// ANCHOR_END: book_hub_1
        assert_eq!(weights.len(), 206);
    }

    #[rustfmt::skip]
    #[test]
    fn book_hub_2() {
        {
// ANCHOR: book_hub_2
use candle::Device;
use hf_hub::api::sync::Api;
use memmap2::Mmap;
use std::fs;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());
let weights_filename = repo.get("model.safetensors").unwrap();

let file = fs::File::open(weights_filename).unwrap();
let mmap = unsafe { Mmap::map(&file).unwrap() };
let weights = candle::safetensors::load_buffer(&mmap[..], &Device::Cpu).unwrap();
// ANCHOR_END: book_hub_2
        assert_eq!(weights.len(), 206);
    }

    // #[rustfmt::skip]
    // #[test]
    // fn book_hub_3() {
    {
// ANCHOR: book_hub_3
use candle::{DType, Device, Tensor};
use hf_hub::api::sync::Api;
use memmap2::Mmap;
use safetensors::slice::IndexOp;
use safetensors::SafeTensors;
use std::fs;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());
let weights_filename = repo.get("model.safetensors").unwrap();

let file = fs::File::open(weights_filename).unwrap();
let mmap = unsafe { Mmap::map(&file).unwrap() };

// Use safetensors directly
let tensors = SafeTensors::deserialize(&mmap[..]).unwrap();
let view = tensors
    .tensor("bert.encoder.layer.0.attention.self.query.weight")
    .unwrap();

// We're going to load shard with rank 1, within a world_size of 4
// We're going to split along dimension 0 doing VIEW[start..stop, :]
let rank = 1;
let world_size = 4;
let dim = 0;
let dtype = view.dtype();
let mut tp_shape = view.shape().to_vec();
let size = tp_shape[0];

if size % world_size != 0 {
    panic!("The dimension is not divisible by `world_size`");
}
let block_size = size / world_size;
let start = rank * block_size;
let stop = (rank + 1) * block_size;

// Everything is expressed in tensor dimension
// bytes offsets is handled automatically for safetensors.

let iterator = view.slice(start..stop).unwrap();

tp_shape[dim] = block_size;

// Convert safetensors Dtype to candle DType
let dtype: DType = dtype.try_into().unwrap();

// TODO: Implement from_buffer_iterator so we can skip the extra CPU alloc.
let raw: Vec<u8> = iterator.into_iter().flatten().cloned().collect();
let tp_tensor = Tensor::from_raw_buffer(&raw, dtype, &tp_shape, &Device::Cpu).unwrap();
// ANCHOR_END: book_hub_3
        assert_eq!(view.shape(), &[768, 768]);
        assert_eq!(tp_tensor.dims(), &[192, 768]);
    }
}

    #[allow(unused)]
    #[rustfmt::skip]
    fn book_training_1() -> Result<()>{
// ANCHOR: book_training_1
use hf_hub::{api::sync::Api, Repo, RepoType};

let dataset_id = "mnist".to_string();

let api = Api::new()?;
let repo = Repo::with_revision(
    dataset_id,
    RepoType::Dataset,
    "refs/convert/parquet".to_string(),
);
let repo = api.repo(repo);
let test_parquet_filename = repo.get("mnist/test/0000.parquet")?;
let train_parquet_filename = repo.get("mnist/train/0000.parquet")?;
let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)?;
let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)?;
// ANCHOR_END: book_training_1
// Ignore unused
let _train = train_parquet;
// ANCHOR: book_training_2
for row in test_parquet {
    for (idx, (name, field)) in row?.get_column_iter().enumerate() {
        println!("Column id {idx}, name {name}, value {field}");
    }
}
// ANCHOR_END: book_training_2
let test_parquet_filename = repo.get("mnist/test/0000.parquet")?;
let train_parquet_filename = repo.get("mnist/train/0000.parquet")?;
let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)?;
let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)?;
// ANCHOR: book_training_3

let test_samples = 10_000;
let mut test_buffer_images: Vec<u8> = Vec::with_capacity(test_samples * 784);
let mut test_buffer_labels: Vec<u8> = Vec::with_capacity(test_samples);
for row in test_parquet{
    for (_name, field) in row?.get_column_iter() {
        if let parquet::record::Field::Group(subrow) = field {
            for (_name, field) in subrow.get_column_iter() {
                if let parquet::record::Field::Bytes(value) = field {
                    let image = image::load_from_memory(value.data()).unwrap();
                    test_buffer_images.extend(image.to_luma8().as_raw());
                }
            }
        }else if let parquet::record::Field::Long(label) = field {
            test_buffer_labels.push(*label as u8);
        }
    }
}
let test_images = (Tensor::from_vec(test_buffer_images, (test_samples, 784), &Device::Cpu)?.to_dtype(DType::F32)? / 255.)?;
let test_labels = Tensor::from_vec(test_buffer_labels, (test_samples, ), &Device::Cpu)?;

let train_samples = 60_000;
let mut train_buffer_images: Vec<u8> = Vec::with_capacity(train_samples * 784);
let mut train_buffer_labels: Vec<u8> = Vec::with_capacity(train_samples);
for row in train_parquet{
    for (_name, field) in row?.get_column_iter() {
        if let parquet::record::Field::Group(subrow) = field {
            for (_name, field) in subrow.get_column_iter() {
                if let parquet::record::Field::Bytes(value) = field {
                    let image = image::load_from_memory(value.data()).unwrap();
                    train_buffer_images.extend(image.to_luma8().as_raw());
                }
            }
        }else if let parquet::record::Field::Long(label) = field {
            train_buffer_labels.push(*label as u8);
        }
    }
}
let train_images = (Tensor::from_vec(train_buffer_images, (train_samples, 784), &Device::Cpu)?.to_dtype(DType::F32)? / 255.)?;
let train_labels = Tensor::from_vec(train_buffer_labels, (train_samples, ), &Device::Cpu)?;

let mnist = candle_datasets::vision::Dataset {
    train_images,
    train_labels,
    test_images,
    test_labels,
    labels: 10,
};

// ANCHOR_END: book_training_3
assert_eq!(mnist.test_images.dims(), &[10_000, 784]);
assert_eq!(mnist.test_labels.dims(), &[10_000]);
assert_eq!(mnist.train_images.dims(), &[60_000, 784]);
assert_eq!(mnist.train_labels.dims(), &[60_000]);
Ok(())
    }
}
