# Input data
hf <- read.csv("heart_failure.csv")

# Load required libraries
library(ggplot2) 
library(caret)
library(randomForest) 
library(xgboost) 
library(pROC) 

# Set seed for reproducibility
set.seed(123)

# Preprocessing
hf <- hf %>%
  mutate(across(c(anaemia, diabetes, high_blood_pressure, sex, smoking, fatal_mi), as.factor))

hf$fatal_mi <- factor(hf$fatal_mi, levels = c(0, 1), labels = c("No", "Yes"))

# Check class balance
table(hf$fatal_mi)

# Check missing values
sum(is.na(hf))  #no missing values

# Summary statistics
str(hf)
summary(hf)

# Histogram of Age Distribution
ggplot(hf, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Age Distribution of Patients",
       x = "Age (years)",
       y = "Frequency") +
  theme_minimal()

# Bar Plot of Sex by Fatal MI Outcome
ggplot(hf, aes(x = sex, fill = fatal_mi)) +
  geom_bar(position = "dodge") +
  labs(title = "Fatal MI Outcome by Sex",
       x = "Sex",
       y = "Count",
       fill = "Fatal MI") +
  theme_minimal()

# Boxplot of Serum Creatinine by Fatal MI Outcome
ggplot(hf, aes(x = fatal_mi, y = serum_creatinine, fill = fatal_mi)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Distribution of Serum Creatinine by Fatal MI Outcome",
       x = "Fatal MI Outcome",
       y = "Serum Creatinine (mg/dL)") +
  theme_minimal()

# Scatter Plot: Ejection Fraction vs Serum Creatinine
ggplot(hf, aes(x = ejection_fraction, y = serum_creatinine, color = fatal_mi)) +
  geom_point(alpha = 0.7) +
  labs(title = "Ejection Fraction vs Serum Creatinine",
       x = "Ejection Fraction (%)",
       y = "Serum Creatinine (mg/dL)",
       color = "Fatal MI") +
  theme_minimal()

# Split data
inTrain <- createDataPartition(hf$fatal_mi, p = 0.7, list = FALSE)
trainData <- hf[inTrain, ]
testData <- hf[-inTrain, ]

# Set up training control for 10-fold cross-validation.
train_control <- trainControl(method = "cv",
                              number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              savePredictions = "final")

# MODEL 1: Logistic Regression
log_model <- train(fatal_mi ~ .,
                   data = trainData,
                   method = "glm",
                   family = "binomial",
                   trControl = train_control,
                   metric = "ROC")

# Evaluate logistic regression performance on test data
pred_log <- predict(log_model, newdata = testData)
cm_log <- confusionMatrix(pred_log, testData$fatal_mi)
print("Logistic Regression Performance:")
print(cm_log)

# MODEL 2: Random Forest
# Tune grid for random forest
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8))

rf_model <- train(fatal_mi ~ .,
                  data = trainData,
                  method = "rf",
                  trControl = train_control,
                  tuneGrid = rf_grid,
                  metric = "ROC")

pred_rf <- predict(rf_model, newdata = testData)
cm_rf <- confusionMatrix(pred_rf, testData$fatal_mi)
print("Random Forest Performance:")
print(cm_rf)

# MODEL 3: XGBoost
# Define grid for tuning XGBoost hyperparameters
xgb_grid <- expand.grid(nrounds = c(50, 100),
max_depth = c(3, 6),
eta = c(0.1, 0.3), 
gamma = c(0, 1),
colsample_bytree = c(0.7, 1),
min_child_weight = 1,
subsample = 1)

xgb_model <- train(fatal_mi ~ .,
data = trainData,
method = "xgbTree",
trControl = train_control,
tuneGrid = xgb_grid,
metric = "ROC")

pred_xgb <- predict(xgb_model, newdata = testData)
cm_xgb <- confusionMatrix(pred_xgb, testData$fatal_mi)
print("XGBoost Performance:")
print(cm_xgb)

# Model Comparison and Selection
results <- resamples(list(Logistic = log_model,
                          RF = rf_model,
                          XGBoost = xgb_model))
summary(results)
dotplot(results)