# Functions to accompany our analysis

#_________________________________________________________________________
# Determining and Plotting the Data Types for all columns
# Source:
# https://stackoverflow.com/questions/21125222/determine-the-data-types-of-a-data-frames-columns

col_missing_function <- function(input_data) {
  # row_missing_function(): a function that finds the % of missing value in each column of the data
  # input_data: a data frame
  
  na_count_col <- apply(input_data, 2, function(y) length(which(is.na(y)))/nrow(input_data)) # counting nas in each column
  na_count_col <- data.frame(na_count_col) #transforming this into a data_frame
  return(na_count_col)
}

dataTypes <- function(input_data) {
  # dataTypes(): a function that finds the data type for each variable in the data
  # input_data: a data frame
  
  col.classes <- lapply(input_data, class)
  col.classes.input_data <- unlist(col.classes) %>%  data.frame()
  temp.data <- table(col.classes.input_data) %>% sort(decreasing=T) %>% data.frame()
  return(temp.data)
}

detect_terms <- function(x, term){
  # detect_terms(): a function that is used to detect a term in each row of the data
  # x: a row in the data
  # term: a term needs to detect
  
  result <- sapply(x, function(y) any(gsub(" ", "", strsplit(y, ",")[[1]])==term))
  if (all(is.na(result))) return(NA)
  result <- ifelse(is.na(result), FALSE, result)
  if (any(result==TRUE)) return("Y")
  return("N")
}

DOE_function <- function(ii, design_scenario, all_data, y, all_ID, holdout_ID_index, all_features, return.prediction=FALSE, seed) {
  # DOE_function(): a function that applies the data to the model specified in the design_scenario
  # ii: index used to navigate to different scenario
  # design_scenario: a data frame that contains all scenarios
  # all_data: a list of all datasets
  # y: the response variable that we want to keep in the data
  # all_ID: a list of all sample IDs for the datasets
  # holdout_ID_index: a list allows to find the corresponding holdout data
  # all_features: a list of features based on the feature selection methods
  # return.prediction: TRUE returns the predicted probabilities for both training and holdout data
  # seed: random seed
  
  # since this function is usually used with a parallel algorithm, we include all 
  # necessary packages
  
  if(require(pacman)==FALSE) install.packages("pacman") # needs to be installed first
  # p_load is equivalent to combining both install.packages() and library()
  pacman::p_load(AUC, caret, conflicted, DMwR, dplyr, e1071, gbm, kernlab, MASS, naivebayes, ranger, ROSE, xgboost)
  conflict_prefer("select", "dplyr")
  
  scenario <- design_scenario[ii,]
  set.seed <- seed
  
  # obtain the necessary information regarding the scenario
  fold <- scenario$fold
  data_scenario <- paste(scenario$imputation_num, scenario$imputation_cat, scenario$encoding, sep="_")
  features <- all_features[[scenario$feature_selection]][[fold]][[data_scenario]]
  algorithm <- scenario$algorithm
  subsampling <- scenario$subsampling
  ID_index <- holdout_ID_index[[fold]]
  input_data <- all_data[[data_scenario]]
  Holdout_ID <- all_ID[[data_scenario]][ID_index[[data_scenario]][1]:ID_index[[data_scenario]][2]]
  
  # obtain the training and hold_out datasets
  training_data <- input_data[!input_data$ID%in%Holdout_ID,] %>% select(c(features, y))
  hold_out <- input_data[input_data$ID%in%Holdout_ID,] %>% select(c(features, y))
  
  # make sure the response variable is categorical
  training_data[[y]] <- as.factor(training_data[[y]])
  levels(training_data[[y]]) <- list(Survived="Survived", Died="Died")
  hold_out[[y]] <- as.factor(hold_out[[y]])
  levels(hold_out[[y]]) <- list(Survived="Survived", Died="Died")
  
  # define the formula that we will use in the modeling
  formul <- as.formula(paste0(as.character(y), "~."))
  
  # decide subsampling method used in the modeling 
  if (subsampling=="none"){
    sampling.method <- NULL
  }else{
    sampling.method <- subsampling
  }
  
  # define g-mean function
  gmeanfunction <- function(data, lev = NULL, model = NULL) {
    sub_sens<-caret::sensitivity(data$pred, data$obs)
    sub_spec<-caret::specificity(data$pred, data$obs)
    return(c(gmean = sqrt(sub_sens*sub_spec)))
  }
  
  # 5 cross validation is used
  control_setting <- caret::trainControl(method = "cv", number = 5, sampling = sampling.method, verboseIter = TRUE,
                                         search ="random", classProbs = TRUE, selectionFunction = "best",
                                         summaryFunction = gmeanfunction)
  
  # decide the method used in the train function
  find_train_method <- function(input_algorithm){
    All_algorithms <- c("ANN", "Bagging", "DT", "ElasticNet", "KPLSR", "LDA", "LR", "NB", "RF", "SVM", "XGB")
    All_methods <- c("nnet", "treebag", "rpart", "glmnet", "kernelpls", "lda", "glm", "naive_bayes", "ranger", "svmRadial", "xgbDART")
    method <- All_methods[which(All_algorithms==input_algorithm)]
    return(method)
  }
  
  alg_method <- find_train_method(algorithm)
  scenario$encoding <- ifelse(scenario$encoding=="OneHot", "One-Hot", scenario$encoding)
  # train the model
  Success <- 1
  
  if (alg_method%in% c("glm", "kernelpls", "nnet", "svmRadial")){
    if((class(try( result_model <- train(formul, data=training_data, method=alg_method, family="binomial",
                                         trControl = control_setting, metric="gmean"), silent = TRUE))=="try-error")[1]){Success<-0}
  }else if (alg_method%in%c("earth", "gbm", "glmnet", "naive_bayes", "ranger", "rpart", "xgbDART", "xgbTree")){
    if((class(try( result_model <- train(formul,  data=training_data, method=alg_method,
                                         trControl = control_setting, tuneLength=10, metric="gmean"),silent = TRUE))=="try-error")[1]){Success<-0}
  }else if(alg_method=="lda"){
    if((class(try(  result_model <- train(formul, data=training_data, method=alg_method, preProcess="pca", preProcOptions = list(method="BoxCox"),
                                          trControl = control_setting, metric="gmean"),silent = TRUE))=="try-error")[1]){Success<-0}
  }else if(alg_method=="treebag"){
    if((class(try(  result_model <- train(formul, data=training_data, method=alg_method, family="binomial",
                                          trControl = control_setting, tuneLength=10, metric="gmean"),silent = TRUE))=="try-error")[1]){Success<-0}
  }
  
  if (Success==1){
    Predicted_Probability <- rep(list(NA), 2)
    names(Predicted_Probability) <- c("training_data", "holdout_data")
    Predicted_Probability[["holdout_data"]] <- as.data.frame(matrix(NA, nrow=nrow(hold_out), ncol=3))
    colnames(Predicted_Probability[["holdout_data"]]) <- c(y, algorithm,"probability")
    Predicted_Probability[["holdout_data"]][,algorithm] <- predict(result_model, newdata=hold_out, type="raw")
    Predicted_Probability[["holdout_data"]][,"probability"] <- predict(result_model, newdata=hold_out, type="prob")[,2]
    
    # to simplify the code, we define the column "probability" for the holdout_data Predicted_Probability as P
    # R as the predicted values for the response variable
    P <- Predicted_Probability[["holdout_data"]][,"probability"]
    R <- Predicted_Probability[["holdout_data"]][,algorithm]
    
    Auc <- AUC::auc(roc(P, hold_out[[y]]))
    Sen <- caret::sensitivity(R, hold_out[[y]])
    Spec <- caret::specificity(R, hold_out[[y]])
    Accu <- (as.data.frame(confusionMatrix(R, hold_out[[y]])$overall))[1,]
    G_mean <- sqrt(Sen*Spec)
    
    Result <- unname(c(unlist(scenario), Auc, Sen, Spec, Accu, G_mean))
    if (return.prediction==FALSE){
      return(Result)
    }else{
      Predicted_Probability[["training_data"]] <- as.data.frame(matrix(NA, nrow=nrow(training_data), ncol=3))
      colnames(Predicted_Probability[["training_data"]]) <- c(y, algorithm,"probability")
      Predicted_Probability[["training_data"]][,y] <- training_data[,y]
      Predicted_Probability[["training_data"]][,algorithm] <- predict(result_model, newdata=training_data, type="raw")
      Predicted_Probability[["training_data"]][,"probability"] <- predict(result_model, newdata=training_data, type="prob")[,2]
      Predicted_Probability[["holdout_data"]][,y] <- hold_out[,y]
      
      return(list(Result, Predicted_Probability))
    }
  }else{
    return(c(unname(unlist(scenario)), NA, NA, NA, NA, NA))
  }
}


dropLowVar <- function(input_data, percent) {
  # dropLowVar(): a function that drops categorical variables that with less variation
  # input_data: a data frame
  # percent: the maximum relative frequency (out of non-missing values) is higher that this value
  
  charColsToKeep <-  input_data[, sapply(input_data, class)=="character"] %>% 
    discard(~ max(table(.x))/sum(!is.na(.x)) * 100 >= percent) %>% colnames()
  nonCharColsToKeep = input_data[, sapply(input_data, class)!="character"] %>% names()
  input_data %<>% select(c(all_of(charColsToKeep), all_of(nonCharColsToKeep)) ) %>% select(sort(names(.)))
  return(input_data)
}


dropNumericMissing <- function(input_data, percent) {
  # dropNumericMissing(): a function that drops numerical variables containing too many missing variables
  # input_data: a data frame
  # percent: the % of missing value is higher that this value, we drop this variable
  
  numColsToKeep <-  input_data[, sapply(input_data, class)=="numeric"] %>% 
    discard(~ sum(is.na(.x))/length(.x) * 100 >= percent) %>% colnames()
  nonNumColsToKeep = input_data[, sapply(input_data, class)!="numeric"] %>% names()
  input_data %<>% select(c(all_of(numColsToKeep), all_of(nonNumColsToKeep)) ) %>% select(sort(names(.)))
  return(input_data)
}


encode_category <- function(input_data, method) {
  # encode_category(): a function that encodes categorical variables
  # input_data: a data frame
  # method: Label or OneHot
  
  input_data %<>% mutate_all(as.factor)
  if (method=="Label"){
    input_data %<>% mutate_all(as.numeric) %>% mutate_all(as.factor) %>% mutate_all(droplevels) 
  }else if (method=="Onehot"){
    numLevels <- apply(input_data, 2, n_distinct)
    dependence <- cumsum(unname(numLevels))
    input_data <- one_hot(data.table(input_data), dropCols = TRUE)
    input_data %<>% select(!all_of(dependence)) %>% mutate_all(as.factor) 
  }
  return(input_data)
}


Get_holdout_index <- function(ID) {
  # Get_holdout_index(): a function that returns a list of intervals that shows
  # the indices (a,b) for the holdout data corresponding to ID[a,b]
  # ID: a list that contains ID for several datasets
  
  k <- unlist(lapply(ID, length))
  I <- sapply(round(k/5), function(x) x*seq(0,4))
  ID_data <- data.frame(rbind(I, k))
  row.names(ID_data) <- NULL
  ID_index <- rep(list(NULL), 5)
  for (i in 1:5){
    ID_index[[i]] <- lapply(ID_data, function(x) c((x[i]+1), x[(i+1)]))
  }
  return(ID_index)
}


recodeGSTATUS <- function(gStatus, gTime, targetYear = 1) {
  # recodeGSTATUS(): a function that decides the survival status at targetYear
  # gStatus: a binary vector 
  # gTime: a continuous vector regarding time
  # targetYear: target year
  
  gStatus %<>% as.numeric()
  gTime %<>% as.numeric()
  response <- ifelse(gTime > 365*targetYear, 'Survived', ifelse(gStatus==0, NA, 'Died') )
  return(response)
}


row_missing_function <- function(input_data) {
  # row_missing_function(): a function to find the % of missing value in each row of the data
  # input_data: a data frame
  
  na_count_row <- apply(input_data, 1, function(y) length(which(is.na(y)))/ncol(input_data)) # counting nas in each row
  na_count_row <- data.frame(na_count_row) #transforming this into a data_frame
  return(na_count_row)
}


search_complete <- function(input_data, max.iter=100000) {
  # search_complete(): a function for a heuristic algorithm which drops rows 
  # and columns based on the percentage of missing values until 
  # just non-empty cells are remained.
  # input_data: a data frame
  # max.iter: the maximum iterations
  
  iter <- 0
  data_temp <- input_data
  while (iter<=max.iter){
    count_col <- col_missing_function(input_data)
    count_row <- row_missing_function(input_data)
    max_col <- max(count_col)
    max_row <- max(count_row)
    max_emp <- max(max_col,max_row)

    if (max_col==0){break()}
    if (max_row==0){break()}
    
    if (max_emp==max_row){
        if((nrow(input_data))>3){
          a <- which(count_row$na_count_row==max_row)
          data_temp <- input_data[-a,]
        }else{
          print("Data have less than 4 rows! The algorithm fails.")
          break()
        }
    }
    
    if (max_col>=max_row){
      b <- names(input_data[which(count_col$na_count_col==max_col)])
      if (max_emp==max_col){
        if ((ncol(input_data))>2){
          #in the next lines, we try to find which columns are the emptiest and then take out the one that I don't want to be deleted
          f <- which(count_col$na_count_col==max_col)
          data_temp <- input_data[, -f]
        }else{
          print("Data have less than 3 columns! The algorithm fails")
          break()
        }
      }
    }else{
      if ((nrow(input_data))>3){
        a <- which(count_row$na_count_row==max_row)
        data_temp <- input_data[-a,]
      }else{
        print("Data have less than 4 rows! The algorithm fails.")
        break()
      }
    }
    iter <- iter + 1
    input_data <- data_temp
  }
  
  if (iter>max.iter){
    print("The algorithm fails. More iterations needed!")
  }else{
    return(data_temp)
  }
}


select_variables <- function(ii, all_data, y, method, ID_all, ID_index, seed=201905) {
  # select_vars(): a function for the variable selection
  # ii: index used to navigate to different data sets 
  # all_data: a list of all datasets
  # y: the response variable that we want to keep in the data
  # method: FFS, LASSO, RF
  # ID_all: a list of all sample IDs for the datasets
  # ID_index: a list allows to find the corresponding holdout data
  # seed: random seed
  
  if(require(pacman)==FALSE) install.packages("pacman")
  pacman::p_load(Biocomb, glmnet, party, ranger)
  
  set.seed <- seed	
  input_data <- as.data.frame(all_data[[ii]])
  all_types <- unlist(lapply(input_data, class))
  char.index <- which(all_types=="factor")
  ID <- ID_all[[ii]][ID_index[[ii]][1]:ID_index[[ii]][2]]
  input_data <- input_data[!input_data$ID%in%ID,]
  input_data[,char.index] <- lapply(input_data[,char.index], droplevels)
  input_data <- input_data[,colnames(input_data)!="ID"]
  input_data[,y] <- as.factor(input_data[,y])
  X <- input_data[,colnames(input_data)!=y]
    
  if (method=="FFS"){
    X$y <- input_data[,y]
    disc <- "MDL"
    threshold <- 0.001
    attrs.nominal <- numeric()
    result_temp <- Biocomb::select.fast.filter(X, disc.method=disc, threshold=threshold, attrs.nominal=attrs.nominal)
    Imp_vars <- as.character(result_temp$Biomarker)
  }else if (method=="LASSO"){
    '%ni%' <- Negate('%in%')
    X <- data.matrix(apply(X, 2, as.numeric))
    glmnet1 <- glmnet::cv.glmnet(x=X,y=as.factor(input_data[,y]),type.measure='auc', family="binomial")
    co <- coef(glmnet1,s = "lambda.1se")
    co <- as.matrix(co[-1,])
    Imp_vars <- row.names(co)[which((co[,1]!=0))]
    Imp_vars <- Imp_vars[Imp_vars %ni% '(Intercept)']
  }else if (method=="RF"){
    Success <- 0
    iteration <- 0
    while (Success==0 & iteration <=10){
          rf <- ranger(formula(paste0(y, "~.")), data=input_data, num.trees = 500, importance = "permutation",
                       scale.permutation.importance = FALSE,  mtry = floor(0.2 * ncol(X)),
                       min.node.size = floor(0.1 * nrow(X)),
                       num.threads = 1,
                       write.forest = TRUE)
          if (min(importance(rf))<0){
            Imp_values <- importance(rf)/abs(min(importance(rf)))
            Imp_vars <- names(which(Imp_values>1))
            Success <- 1
          }else{
            iteration <- iteration + 1
          }
    }
    if (iteration>10){
      print("At least 10 runs with no negative importance score!")
      Imp_vars <- NA
    }
  }
  output <- list(Imp_vars)
  names(output) <- names(ID_all)[ii]
  return(output)
}



