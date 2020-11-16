library(devtools)
library(tensorflow)
library(keras)
getwd()
setwd("FoodCNN")
model <- load_model_hdf5("food101.hdf5")

test_dir <- "test//"

food101Data <- list.files(path = test_dir,
                            full.names = FALSE, recursive = TRUE)

test_datagen <- image_data_generator(rescale = 1/255)  


test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(224, 224),
  batch_size = 20,
  class_mode = "categorical"
)

model %>% evaluate_generator(test_generator, steps = 50)
