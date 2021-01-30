#By Joshua Ball
#Reads in a single image, calculates the PCA and saves that image
library(parallel)
cl <- parallel::makePSOCKcluster(11)#Utilizing 11 of 12 cores on my computer
doParallel::registerDoParallel(cl)
library(foreach)
library(tidyr)
library(jpeg)
#Used to save images easily.
library(imager)  
#Used to filter noise in images
library(EBImage)
library(wvtool)
directoryToImages <- "images//"

setwd("C:\\Users\\JoshPC\\Desktop\\food-101")

#class_labels <- read.delim("meta//classes.txt")
food101Data <- list.files(path = directoryToImages,
                          full.names = TRUE, recursive = TRUE)
data <- data.frame
data <- food101Data
head(data)
ptm1 <- proc.time()

foreach(x = 1:length(data), .packages=c("jpeg","imager","EBImage","magick") ) %dopar% {
  
  #load in image
  cur_image <- readJPEG(data[x])
  print(paste("Current image:", data[x]))
  
  #Read each color layer of image
  r <- cur_image[,,1]
  g <- cur_image[,,2]
  b <- cur_image[,,3]
  #Do the seperate PCA for each layer
  cur_image.r.pca <- prcomp(r, center = FALSE)
  cur_image.g.pca <- prcomp(g, center = FALSE)
  cur_image.b.pca <- prcomp(b, center = FALSE)
  #combine PCA's into a single list
  rgb.pca <- list(cur_image.r.pca, cur_image.g.pca, cur_image.b.pca)
  #Lets set our number of componets to a hard number here
  number_of_componets = 220
  #apply our pca to the image
  pca.img <- sapply(rgb.pca, function(j) {
    compressed.img <- j$x[,1:number_of_componets] %*% t(j$rotation[,1:number_of_componets])
  }, simplify = 'array')
  #Write the image
  #writeJPEG(pca.img, paste('compressed', round(number_of_componets,0), '_components.jpg', sep = ''))
  
  finale <- image_read(pca.img)
  actual_finale <- magick2cimg(finale)
  save.image(actual_finale, file = gsub("JPG", "jpg", paste(data[x])))
  
  sink("Report_V2_PCA.txt", append=TRUE)
  cat("Time taken", proc.time() - ptm1)
  
}

