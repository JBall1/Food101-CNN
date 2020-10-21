library(devtools)

library(tensorflow)

library(keras)


#MARK: 
#     1. Create train and test sets, save them to file

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

#75/25 split
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

#Read in directories 
train_dir <- "FoodCNN/train/"
test_dir <- "FoodCNN/test/"
#Setup test, train and validation data generators with data augmentation. 
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

#Use transfer learning with the MobilenetV2 architecture and imagenet weights
conv_base <- application_mobilenet_v2(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)
summary(conv_base)


#1. We add our mobilenetV2 model on the 'top' our of model to handle the heavy learning. If you need more info about transfer learning please ask.
#2. Global average pooling 2D essentially takes a block of a tensor (HxWxD) into a reduced size 1x1xD dimension object and takes the average of H and W.
#   Really important for image classification. 
#3. Softmax activation distributes the probability throughout each output node. Ideal for categorical data like ours.
#4. We are using an l2 regulizer, or Ridge Regression, to help ensure we do not overfit. It essentially adds a square(^2) coefficent as a 'penalty term' to our loss.
#5. A general note: you must be careful in how much you 'punish' a model for 'trying' to overfit, as it may not learn at all! Ours does though :)
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.2) %>%
  layer_dense(activation = "softmax",101,kernel_regularizer = regularizer_l2(0.01)) #%>%

#monitoring accuracy while training, just something you can pick. 
#We are solving a multiclass problem, so cateogrical loss is needed. Cateogrical crossentropy because it has great performance and provides our labels as one-hot
#We are using adam as our optimizer, a first order stochastic gradient descent method, because it is basically the best one out there in genral. With some tuning,
#like we have done, tuning the lr(learning rate) to 1e-3.
model %>% compile(
  loss="categorical_crossentropy",
  optimizer=optimizer_adam(lr=1e-3),
  metrics = c("accuracy")
)

#Setting up checkpoints 
checkpoint_dir <- "FoodCNN/checkpoints/"
#Our models automatically saved with the following name, including its val loss and val acc.
checkpoint_name <- paste(checkpoint_dir, "food101-{val_loss:.4f}-{val_acc:.4f}.hdf5")
#For early stopping, model saving
#Early stopping allows our model to stop training if val_acc hasn't imporved for 18 epochs
#model checkpoint saves our model if the val_acc has improved
#reduce lr reduces the learning rate if our val_loss hasn't improved for 10 epochs.
my_callbacks <- list(callback_early_stopping(monitor = "val_acc", patience = 18, verbose = 1),
                     callback_model_checkpoint(checkpoint_name,monitor = "val_acc",verbose = 1, save_best_only = TRUE),
                     callback_reduce_lr_on_plateau(monitor = "val_loss",factor = 0.1,
                                                   patience = 10,min_lr = 1e-4))
  
#Training our model
#Change workers to number of threads you have.(2 is a good base point if you don't know).
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 25,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = length(validation_generator),
  workers = 6,
  callbacks = my_callbacks
)

