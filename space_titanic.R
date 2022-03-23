###################
#     SETUP       #
###################

library(tidyverse)   # Trousse à outils générique
library(caret)       # Apprentissage machine
library(DataExplorer)   # Résumé dataset
library(mice)        # Gestion valeurs manquantes
library (GGally)     # Dual plots
memory.limit(size = 50000)

# Télécharger données
datazip <- tempfile()
#download.file("https://github.com/EKRihani/space_titanic/raw/main/spaceship_titanic.zip", datazip)
datazip <- "~/projects/space_titanic/spaceship_titanic.zip"
train_set <- unzip(datazip, "train.csv")
train_set <- read.csv(train_set, header = TRUE, sep = ",")
test_set <- unzip(datazip, "test.csv")
test_set <- read.csv(test_set, header = TRUE, sep = ",")
# OSEF
# result_set <- unzip(datazip, "sample_submission.csv")
# result_set <- read.csv(result_set, header = TRUE, sep = ",")

# Introduction dataset
head(train_set)
str(train_set)
# create_report(data = train_set, y = "Transported")

# Ajustement classes
train_set$HomePlanet <- as.factor(train_set$HomePlanet)
train_set$CryoSleep <- as.factor(train_set$CryoSleep)
train_set$Destination <- as.factor(train_set$Destination)
train_set$VIP <- as.factor(train_set$VIP)
train_set$Transported <- as.factor(train_set$Transported)
test_set$HomePlanet <- as.factor(test_set$HomePlanet)
test_set$CryoSleep <- as.factor(test_set$CryoSleep)
test_set$Destination <- as.factor(test_set$Destination)
test_set$VIP <- as.factor(test_set$VIP)

# # Extraction de features PRÉ-MICE
# train_set$Cabin1 <- str_split(string = train_set$Cabin, pattern = "/", simplify = TRUE)[,1]
# train_set$Cabin2 <- str_split(string = train_set$Cabin, pattern = "/", simplify = TRUE)[,2]
# train_set$Cabin3 <- str_split(string = train_set$Cabin, pattern = "/", simplify = TRUE)[,3]
# train_set$Group <- str_split(string = train_set$PassengerId, pattern = "_", simplify = TRUE)[,1]
# train_set$SpendRSD <- train_set$RoomService + train_set$VRDeck + train_set$Spa
# train_set$SpendMF <- train_set$FoodCourt + train_set$ShoppingMall
# train_set$Spending <- train_set$SpendMF + train_set$SpendRSD
# test_set$Cabin1 <- str_split(string = test_set$Cabin, pattern = "/", simplify = TRUE)[,1]
# test_set$Cabin2 <- str_split(string = test_set$Cabin, pattern = "/", simplify = TRUE)[,2]
# test_set$Cabin3 <- str_split(string = test_set$Cabin, pattern = "/", simplify = TRUE)[,3]
# test_set$Group <- str_split(string = test_set$PassengerId, pattern = "_", simplify = TRUE)[,1]
# test_set$SpendRSD <- test_set$RoomService + test_set$VRDeck + test_set$Spa
# test_set$SpendMF <- test_set$FoodCourt + test_set$ShoppingMall
# test_set$Spending <- test_set$SpendRSD + test_set$SpendMF
# 
# create_report(data = train_set, y = "Transported")

# Définir fonction : trace distributions selon le critère principal
plot_stuff <- function(fct_dataset, fct_criterion){
   structure <- sapply(X = fct_dataset, FUN = class, simplify = TRUE)
   for (n in 1:ncol(fct_dataset)){
      plot <- train_set %>%
         ggplot(aes_string(x = colnames(fct_dataset)[n], fill = fct_criterion))
      if(as.character(structure[n]) %in% c("integer", "numeric"))
      {plot <- plot + geom_histogram(position = "fill")} # position = "fill"
      else
      {plot <- plot + geom_bar(position = "fill")} # position = "fill"
      plotname <- paste0("plot_", colnames(fct_dataset)[n])
      assign(plotname, plot, envir = .GlobalEnv)
   }
}

# Définir fonction : train/fit (en 5x 10-fold cross-validation)
fit_test <- function(fcn_dataset, fcn_criterion, fcn_factors, fcn_model, fcn_tune){
   tr_ctrl <- trainControl(classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 5)
   cmd <- paste0("train(", fcn_criterion, " ~ ", fcn_factors, ", method = '", fcn_model, "', data = ", fcn_dataset, ", trControl = tr_ctrl, ", fcn_tune,")")
   eval(parse(text = cmd))
}
# Définir fonction : FAST train/fit (en 5-fold cross-validation)
FASTfit_test <- function(fcn_dataset, fcn_criterion, fcn_factors, fcn_model, fcn_tune){
   tr_ctrl <- trainControl(classProbs = TRUE, method = "cv", number = 5)
   cmd <- paste0("train(", fcn_criterion, " ~ ", fcn_factors, ", method = '", fcn_model, "', data = ", fcn_dataset, ", trControl = tr_ctrl, ", fcn_tune,")")
   eval(parse(text = cmd))
}

# Distributions descriptives basiques
plot_stuff(train_set, "Transported")

# Gestion valeurs manquantes
md.pattern(train_set, rotate.names = TRUE)
pred_matrix <- make.predictorMatrix(train_set)
pred_matrix["PassengerId",] <- 0
pred_matrix[,"PassengerId"] <- 0
pred_matrix["Cabin",] <- 0
pred_matrix[,"Cabin"] <- 0
pred_matrix["Name",] <- 0
pred_matrix[,"Name"] <- 0
pred_matrix["Transported",] <- 0
pred_matrix[,"Transported"] <- 0
mice_input <- mice(train_set, method = "rf", predictorMatrix = pred_matrix, m = 10)  # Prédiction MICE sur données entrainement (pmm, midastouch, sample, cart, rf)
train_set <- complete(mice_input)                # Remplissage valeurs manquantes

md.pattern(test_set, rotate.names = TRUE)
pred_matrix <- make.predictorMatrix(test_set)
pred_matrix["PassengerId",] <- 0
pred_matrix[,"PassengerId"] <- 0
pred_matrix["Cabin",] <- 0
pred_matrix[,"Cabin"] <- 0
pred_matrix["Name",] <- 0
pred_matrix[,"Name"] <- 0
mice_input <- mice(test_set, method = "pmm", predictorMatrix = pred_matrix, m = 10)
test_set <- complete(mice_input)

# Extraction de features POST-Mice
train_set$Cabin1 <- str_split(string = train_set$Cabin, pattern = "/", simplify = TRUE)[,1]
train_set$Cabin2 <- str_split(string = train_set$Cabin, pattern = "/", simplify = TRUE)[,2]
train_set$Cabin3 <- str_split(string = train_set$Cabin, pattern = "/", simplify = TRUE)[,3]
train_set$Group <- str_split(string = train_set$PassengerId, pattern = "_", simplify = TRUE)[,1]
train_set$Grp <- floor(as.numeric(train_set$Group)/8)
train_set$SpendRSD <- train_set$RoomService + train_set$VRDeck + train_set$Spa
train_set$SpendMF <- train_set$FoodCourt + train_set$ShoppingMall
train_set$Spending <- train_set$SpendMF + train_set$SpendRSD
test_set$Cabin1 <- str_split(string = test_set$Cabin, pattern = "/", simplify = TRUE)[,1]
test_set$Cabin2 <- str_split(string = test_set$Cabin, pattern = "/", simplify = TRUE)[,2]
test_set$Cabin3 <- str_split(string = test_set$Cabin, pattern = "/", simplify = TRUE)[,3]
test_set$Group <- str_split(string = test_set$PassengerId, pattern = "_", simplify = TRUE)[,1]
test_set$Grp <- floor(as.numeric(test_set$Group)/8)
test_set$SpendRSD <- test_set$RoomService + test_set$VRDeck + test_set$Spa
test_set$SpendMF <- test_set$FoodCourt + test_set$ShoppingMall
test_set$Spending <- test_set$SpendRSD + test_set$SpendMF

create_report(data = train_set, y = "Transported")

plot_stuff(train_set, "Transported")

pair_plots <- ggpairs(
   train_set,
   columns = c(6,7,8,9,10),
   lower = NULL,
   diag = list(continuous = wrap("densityDiag", alpha = .6), 
               discrete = wrap("barDiag")
   ),
   upper = list(continuous = wrap("points", alpha = .3, shape = 20), 
                combo = wrap("dot", alpha = .3, shape = 20),
                discrete = wrap("dot_no_facet", alpha = .3, shape = 20)
   ),
   ggplot2::aes(color = Transported)
)
pair_plots

train_set[1:100,] %>%
   ggplot(aes(x = Group, fill = Transported)) +
   geom_bar(position = "fill")

# KNN
fit_kknn_kmax <- fit_test("train_set", "Transported", "Age + CryoSleep + HomePlanet + VIP + SpendRSD + SpendMF + Cabin3 + Destination", "kknn", "tuneGrid  = data.frame(kmax = round(seq(from = 10, to = 100, length.out = 15)), distance= 2, kernel = 'optimal')")
fit_kknn_distance <- FASTfit_test("train_set", "Transported", "Age + CryoSleep + HomePlanet + VIP + SpendRSD + SpendMF + Cabin3 + Destination", "kknn", "tuneGrid  = data.frame(kmax = 9, distance= 1:6, kernel = 'optimal')")
fit_kknn_kernel <- FASTfit_test("train_set", "Transported", "Age + CryoSleep + HomePlanet + VIP + SpendRSD + SpendMF + Cabin3 + Destination", "kknn", "tuneGrid  = data.frame(kmax = 80, distance=1, kernel = c('triangular', 'epanechnikov', 'biweight', 'triweight', 'gaussian', 'cos', 'inv','rank', 'optimal'))")
fit_kknn_kmax
fit_kknn_distance
fit_kknn_kernel

# Linear Discriminant Analysis
fit_lda <- fit_test("train_set", "Transported", ".", "lda", "")
fit_lda

# Forêt CART
fit_rpart <- fit_test("train_set", "Transported", ".", "rpart2", "tuneGrid=data.frame(maxdepth = 2:3)")
fit_rpart <- fit_test("train_set", "Transported", ".", "rpart", "")
fit_rpart$results
plot(fit_rpart$finalModel) + text(fit_rpart$finalModel)


# Ranger (forêt aléatoire)
fit_ranger_mtry <- fit_test("train_set", "Transported", "Age + CryoSleep + VIP + Spending + HomePlanet + Group", "ranger", "tuneGrid  = data.frame(mtry = round(seq(from = 1, to = 10, length.out = 6)), splitrule = 'extratrees', min.node.size = 2), num.trees = 6")
fit_ranger_splitrule <- fit_test("train_set", "Transported", "Age + CryoSleep + VIP + Spending +Cabin + HomePlanet + Group", "ranger", "tuneGrid  = data.frame(splitrule = c('gini', 'extratrees'), mtry = 50, min.node.size = 2), num.trees = 6")
fit_ranger_nodesize <- fit_test("train_set", "Transported", "Age + CryoSleep + VIP + Spending + HomePlanet + Group", "ranger", "tuneGrid  = data.frame(min.node.size = round(seq(from = 1, to = 20, length.out = 6)), mtry = 4, splitrule = 'extratrees'), num.trees = 6")
fit_ranger <- FASTfit_test("train_set", "Transported", "Age + CryoSleep + VIP + SpendRSD + HomePlanet + Group + Cabin1 + Cabin3", "ranger", "num.trees = 3")
max(fit_ranger$results["Accuracy"])
fit_ranger$bestTune

# Rborist (forêt aléatoire)
# fit_Rborist_pred <- fit_test("train_set", "Transported", ".", "Rborist", "tuneGrid  = data.frame(predFixed = round(seq(from = 1, to = 1000, length.out = 6)), minNode = 2), ntrees = 5")
# fit_Rborist_minNode <- fit_test("train_set", "Transported", ".", "Rborist", "tuneGrid  = data.frame(minNode = 1:5, predFixed =50), ntrees = 5")
fit_Rborist <- fit_test("train_set", "Transported", "Sex + Fare + Pclass", "Rborist", "ntrees = 10")  # BUGGG
max(fit_Rborist$results["Accuracy"])


# GAM Loess
# fit_gamLoess_span <- fit_test("train_set", "Transported", ".","gamLoess", "tuneGrid  = data.frame(span = seq(from = 0.01, to = 0.99, length.out = 5), degree = 1)")
# fit_gamLoess_degree <- fit_test("train_set", "Transported", ".","gamLoess", "tuneGrid  = data.frame(degree = c(0, 1), span = 0.5)")
fit_gamLoess <- fit_test("train_set", "Transported", "Sex + Fare + Pclass", "gamLoess", "")
max(fit_gamLoess$results["Accuracy"])

# Xtreme Gradient Boosting
fit_xgbLinear <- fit_test("train_set", "Transported", "Sex + Fare + Pclass + Age", "xgbLinear", "")
max(fit_xgbLinear$results["Accuracy"])
fit_xgbTree <- fit_test("train_set", "Transported", "Age + CryoSleep + VIP + SpendRSD + HomePlanet + Cabin1 + Cabin3 + Destination", "xgbTree", "")
max(fit_xgbTree$results["Accuracy"])
fit_xgbTree$bestTune
fit_xgbDART <- fit_test("train_set", "Transported", "Age + CryoSleep + VIP + SpendRSD + HomePlanet + Cabin1 + Cabin3 + Destination", "xgbDART", "")
max(fit_xgbDART$results["Accuracy"])
fit_xgbDART$bestTune

# Prediction finale
fit <- train(Transported ~ Age + CryoSleep + VIP + SpendRSD + HomePlanet + Cabin1 + Cabin3 + Destination, method = "xgbDART", data = train_set, 
             tuneGrid  = data.frame(nrounds=100, max_depth=2, eta=.3, gamma=0, subsample=1, colsample_bytree=.6, rate_drop = .5, skip_drop=.95, min_child_weight=1))
fit <- train(Transported ~ Age + CryoSleep + VIP + SpendRSD + HomePlanet + Cabin1 + Cabin3 + Destination, method = "ranger", data = train_set, 
             tuneGrid  = data.frame(mtry=15, splitrule="extratrees", min.node.size=1), num.trees = 6)
fit <- train(Transported ~ Age + CryoSleep + HomePlanet + VIP + SpendRSD + SpendMF + Cabin3 + Destination, method = "knn", data = train_set, 
             tuneGrid  = data.frame(k=30))
fit <- train(Transported ~ Age + CryoSleep + HomePlanet + VIP + SpendRSD + SpendMF + Cabin3 + Destination, method = "kknn", data = train_set, 
             tuneGrid  = data.frame(kmax = 50, distance=2, kernel = "optimal"))


RESULT <- NULL
RESULT$PassengerId <- test_set$PassengerId
RESULT$Transported <- predict(object = fit, newdata = test_set)
RESULT <- as.data.frame(RESULT)
write.csv(RESULT, "result.csv", row.names = FALSE)

