library(devtools)

library(tensorflow)

library(keras)
setwd("C:\\Users\\JoshPC\\Desktop\\food-101")
base_dir <- "C:\\Users\\JoshPC\\Desktop\\food-101"
train_dir <- file.path(base_dir, "train")
test_dir <- file.path(base_dir, "test")
val_dir <- file.path(base_dir, "val")

dir.create(train_dir)
dir.create(test_dir)
dir.create(val_dir)


directoryToImages <- "images//"

food101Data <- list.files(path = directoryToImages,
                          full.names = FALSE, recursive = TRUE)

class_labels <- scan("meta//classes.txt", what="", sep="\n")
data <- data.frame
data <- food101Data

require(caTools)
set.seed(101) 

sample = sample.split(data, SplitRatio = .75)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)



#moving train set files into train dir
for (i in train){
  new_dir <- paste0(train_dir, "/",i)
  old_dir <- paste0("images//",i)
  file.copy(file.path(old_dir), 
            file.path(new_dir))
}
#moving test set files into test dir
for (i in test){
  new_dir <- paste0(test_dir, "/",i)
  old_dir <- paste0("images//",i)
  file.copy(file.path(old_dir), 
            file.path(new_dir))
}





index <- 1:101
par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))

test_datagen <- image_data_generator(rescale = 1/255)  

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

validation_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(150, 150),  # Resizes all images to 150 × 150
  batch_size = 20,
  class_mode = "categorical"       # binary_crossentropy loss for binary labels
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)


conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)
summary(conv_base)
model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")


freeze_weights(conv_base)
length(model$trainable_weights)
unfreeze_weights(conv_base, from = "block3_conv1")

model %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)
plot(history)
