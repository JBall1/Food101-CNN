library(devtools)

library(tensorflow)

library(keras)


#Used to save the various train and test data samples into their 

#base_dir <- "C:\\Users\\JoshPC\\Desktop\\food-101"
#train_dir <- file.path(base_dir, "train")
#test_dir <- file.path(base_dir, "test")
#val_dir <- file.path(base_dir, "val")

#dir.create(train_dir)
#dir.create(test_dir)
#dir.create(val_dir)


#directoryToImages <- "images//"

#food101Data <- list.files(path = directoryToImages,
                          #full.names = FALSE, recursive = TRUE)

#class_labels <- scan("meta//classes.txt", what="", sep="\n")
#data <- data.frame
#data <- food101Data

#require(caTools)
#set.seed(101) 

#sample = sample.split(data, SplitRatio = .75)
#train = subset(data, sample == TRUE)
#test  = subset(data, sample == FALSE)



#moving train set files into train dir
#for (i in train){
 # new_dir <- paste0(train_dir, "/",i)
 #old_dir <- paste0("images//",i)
 # file.copy(file.path(old_dir), 
#            file.path(new_dir))
#}
#moving test set files into test dir
#for (i in test){
#  new_dir <- paste0(test_dir, "/",i)
#  old_dir <- paste0("images//",i)
#  file.copy(file.path(old_dir), 
#            file.path(new_dir))
#}
#END

#Start of CNN creation and training
train_dir <- "FoodCNN/train/"
test_dir <- "FoodCNN/test/"

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
  target_size = c(224, 224),
  batch_size = 64,
  class_mode = "categorical"
)

train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(224, 224),  # Resizes all images to 224 × 224
  batch_size = 64,
  class_mode = "categorical"     
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(224, 224),
  batch_size = 64,
  class_mode = "categorical"
)


conv_base <- application_mobilenet_v2(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)
summary(conv_base)


#Add our top layers for our new data
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.2) %>%
  layer_dense(activation = "softmax",101,kernel_regularizer = regularizer_l2(0.01)) #%>%


model %>% compile(
  loss="categorical_crossentropy",
  optimizer=optimizer_adam(lr=1e-3),
  metrics = c("accuracy")
)

checkpoint_dir <- "FoodCNN/checkpoints/"
checkpoint_name <- paste(checkpoint_dir, "food101-{val_loss:.4f}-{val_acc:.4f}.hdf5")
#for early stopping, model saving
my_callbacks <- list(callback_early_stopping(monitor = "val_acc", patience = 18, verbose = 1),
                     callback_model_checkpoint(checkpoint_name,monitor = "val_acc",verbose = 1, save_best_only = TRUE),
                     callback_reduce_lr_on_plateau(monitor = "val_loss",factor = 0.1,
                                                   patience = 10,min_lr = 1e-4))
  

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 25,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = length(validation_generator),
  workers = 6,
  callbacks = my_callbacks
)
plot(history)

