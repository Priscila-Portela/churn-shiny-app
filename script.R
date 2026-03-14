# PREDICT CUSTOMER CHURN PROJECT


# 1. IMPORT LIBRARIES -----------------------------------------------------

library(readr)
library(rsample)
library(dplyr)
library(tidyr)
library(ggplot2)


set.seed(42)

# 2. READ DATA ------------------------------------------------------------

df <- read_csv('train.csv', show_col_types = FALSE)
df_prod <- read_csv('test.csv', show_col_types = FALSE)

#  3. SAMPLE ----------------------------------------------------------

## 3.1 Features and Target -------------------------------------------------
# Before spliting data into train and test we are going to 
# define the features and the target:
features <- names(df)[2:20]
target <- 'Churn'


## 3.2. Split data ---------------------------------------------------------

# The dataset was divided into training and testing sets using 
# a stratified train-test split to preserve the distribution of 
# the target variable (Churn).
split <- initial_split(df, prop = 0.8, strata = Churn)

train <- training(split)
test  <- testing(split)

#Shows the proportion of Yes/No Churn in the datasets of training and testing
prop.table(table(train$Churn))
prop.table(table(test$Churn))


#  4. Data Description ----------------------------------------------------

## 4.1 Data types ----------------------------------------------------------

train$SeniorCitizen <- as.factor(train$SeniorCitizen)
train$Churn <- as.factor(train$Churn) # garantir que o target seja categorico

## 4.2 Missing values ------------------------------------------------------
colSums(is.na(train))

#No missing values were found


# 5. FEATURE ENGINEERING --------------------------------------------------
# Criamos a variavel que identifica se o pagamento ocorre de forma automatica
train <- train %>% mutate(automatic_payment = ifelse(grepl("automatic", PaymentMethod), 1, 0))


# 6. EXPLORATORY DATA ANALYSIS --------------------------------------------

## 6.1 Identify features -------------------------------------------------
cat_features <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
                  "MultipleLines", "InternetService","OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                  "Contract", "PaperlessBilling", "PaymentMethod", "automatic_payment")

num_features <- c("tenure", "MonthlyCharges", "TotalCharges")

## 6.2  Numerical analysis -------------------------------------------------

train %>%
  select(all_of(num_features), Churn) %>%                                     #seleciona variaveis numericas e a alvo (churn)
  pivot_longer(cols = -Churn, names_to = "variable", values_to = "value") %>% #coloca o dataset em formato longo para o ggplot
  ggplot(aes(x = Churn, y = value, fill = Churn)) +                           #cria o gráfico base
  geom_violin(trim = FALSE) +                                                 #define o gráfico de violino e o trim=false faz mostrar a distribuição completa, sem cortar as extremidades
  facet_wrap(~variable, scales = "free")                                      # divide o gráfico por variável, um gráfico para cada variável. scales=free faz com que cada gráfico tenha sua própria escala

#Gráfico de violino para observar a distribuição das variaveis numericas do dataset de treino
####### Insights:
# 1. Monthly Charges: corresponde ao valor da cobrança mensal. 

# 3. Tenure: corresponde ao tempo como cliente.


##  6.3 Categorical analysis -----------------------------------------------

train %>%
  mutate(across(all_of(cat_features), as.character)) %>%
  pivot_longer(cols = all_of(cat_features),
               names_to = "variable",
               values_to = "category") %>%
  ggplot(aes(x = category, fill = Churn)) +
  geom_bar(position = "fill") +
  facet_wrap(~variable, scales = "free_x") +
  labs(y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("Yes" = "#F8766D", "No" = "#00BFC4"))


## 6.4. Features relevance  ---------------------------------------------------
library(rpart)
library(rpart.plot)

# ajustar o modelo
tree_model <- rpart(
  Churn ~ .,
  data = train,
  method = "class"
)

# visualizar a arvore

rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  cex = 0.8
)

tree_model$variable.importance

