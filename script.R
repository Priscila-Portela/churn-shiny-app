# PREDICT CUSTOMER CHURN PROJECT


# 1. IMPORT LIBRARIES -----------------------------------------------------

library(readr)
library(rsample)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tidyverse)
library(tidytext)
library(fastDummies)
library(randomForest)
library(FSelectorRcpp)
library(caret)

set.seed(42)

# 2. READ DATA ------------------------------------------------------------

df <- read_csv('data/train.csv', show_col_types = FALSE)
df_prod <- read_csv('data/test.csv', show_col_types = FALSE)

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

# 77% dos clientes são no-churn.
# 23% são churn
## 3.3 Identify features -------------------------------------------------
cat_features <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
                  "MultipleLines", "InternetService","OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                  "Contract", "PaperlessBilling", "PaymentMethod")

num_features <- c("tenure", "MonthlyCharges", "TotalCharges")

#  4. DATA PREPARATION ----------------------------------------------------

## 4.1 Rename ------------------------------------------------------
train[train == "No internet service"] <- "NIS"


train[train == "No phone service"] <- "NPS"

## 4.2 Data types ----------------------------------------------------------

# Categorical in factors
for (feature in cat_features) {
  train[[feature]] <- as.factor(train[[feature]])}

## 4.2 Missing values ------------------------------------------------------
colSums(is.na(train))

#No missing values were found



# 5. EXPLORATORY DATA ANALYSIS --------------------------------------------


## 5.1  Numerical analysis -------------------------------------------------

#######  Gráfico de densidade. Representa a proporção relativa de observações naquele valor, sendo o total da área da curva =1.
# Ou seja, não mostra a distribuição de forma absoluta e sim a forma da distribuição.
# Foi optado por este gráfico porque temos classes de churn e no-churn desbalanceadas (23/77).
train %>%
  select(all_of(num_features), Churn) %>% #seleciona variaveis numericas e a alvo (churn)
  pivot_longer(cols = -Churn, names_to = "variable", values_to = "value") %>% #coloca o dataset em formato longo para o ggplot
  ggplot(aes(x = value, fill = Churn)) + #cria o gráfico base
  geom_density(alpha = 0.4) + # Transparency rate
  facet_wrap(~variable, scales = "free") + # divide o gráfico por variável, um gráfico para cada variável. scales=free faz com que cada gráfico tenha sua própria escala
  scale_fill_manual(values = c("Yes" = "#F8766D", "No" = "#00BFC4")) # configura cores
####### Insights:
# 1. Monthly Charges: corresponde ao valor da cobrança mensal. Muitos clientes que pagam menos de 25 por mês em suas faturas, optam pro não cancelar os seus serviços.
# Grande parte dos clientes que cancelam suas assinaturas, pagam entre 70 e 110 mensalmente.

# 2. Tenure: corresponde ao tempo como cliente. A maioria dos clientes churn são clientes a menos de 20 meses. 
# Há uma grande quantidade de clientes fidelizados com mais de 70 meses de contrato.

# 3. TotalCharges: representa o total já pago pelo cliente para empresa (soma acumulada de todas as faturas). 
# Clientes que cancelaram o produto tem um valor acumulado pago menor que 2500, assim como a maioria dos atuais clientes
# Não é uma boa variável para prever churn, pois em geral, ambos os grupos de churn e no-churn tem uma distribuição similar.


##  5.2 Categorical analysis -----------------------------------------------

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

####### Insights:
# 1. contract: Muitos clientes que pagam mês a mês cancelam seus contratos. contratos de um ou dois anos tendem á favorecer a fidelidade do cliente.
# 2. Dependents: Clientes com dependentes são mais fiéis.
# 3. Device Protection: há um maior número de clientes churn que não possuiam Device Protection.
# 4. Gender: não há diferença observável entre o genero de clientes.
# 5. Internet Service: Clientes com fibra ótica costumam cancelar mais os serviços.
# 5.1. Hipótese: clientes com DSL são clientes mais antigos, portanto são clientes que são mais fiéis?
# 6. Multiple lines: clientes com multiplas linhas tem maior taxa de cancelamento, apesar de ter uma diferença discreta entre as classes.
# 7. Online Backup/Security: clientes que não assinam estes serviços tem maior taxa de churn.
# 8. paperless billing: maior taxa de cancelamento em clientes que recebem suas contas de forma online.
# 9. Partner: Clientes sem partner tem maior tendencia a cancelar o serviço.
# 10. Payment Method: Clientes que utilizam o meio de pagamento eletrônico não automático tem maior probabilidade de cancelar a assinatura, comparado aos demais métodos de pagamento.
# 11. Senior Citizen: clientes jóvens são mais fiéis.
# 12. Streaming movies/TV: Não há muita correlação entre assinantes de streaming de filmes/TV e cancelamento dos serviços.
# 13. Tech Suport: Clientes que não possuem o suporte técnico tendem a cancelar mais os serviços.



# 5.3. General data category distribution ---------------------------------
## Objetivo: verificar a distribuição dos dados em cada categoria, independente de ser churn ou no-churn
## Objetivo: verificar a ocorrência de categorias raras


train %>%
  select(where(is.factor)) %>% 
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "category"
  ) %>%
  count(variable, category) %>%
  group_by(variable) %>%
  mutate(
    percent = n / sum(n),
    limit = max(n) * 0.2
  ) %>%
  
  ggplot(aes(x = n, y = reorder_within(category, n, variable))) +
  
  geom_col(fill = "grey40") +
  
  geom_text(
    data = ~ dplyr::filter(.x, n >= limit),
    aes(label = scales::percent(percent, accuracy = 0.1)),
    color = "white",
    size = 3,
    hjust = 1
  ) +
  
  geom_text(
    data = ~ dplyr::filter(.x, n < limit),
    aes(label = scales::percent(percent, accuracy = 0.1)),
    color = "black",
    size = 3,
    hjust = -0.2
  ) +
  
  facet_wrap(~variable, scales = "free", ncol = 5, nrow = 5) +
  
  scale_y_reordered() +
  
  labs(x = NULL, y = NULL) +
  
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  
  theme_void() +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 8, hjust=1),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank()
  ) 


# 6. FEATURE ENGINEERING --------------------------------------------------
# Criamos a variavel que identifica se o pagamento ocorre de forma automatica
train <- train %>% mutate(automatic_payment = ifelse(grepl("automatic", PaymentMethod), 1, 0),) 
train$automatic_payment <- as.factor(train$automatic_payment)

# 7. ENCODING -------------------------------------------------------------
## 7.1. Label/Binary Encoding ---------------------------------------------------
## features: Dependents, gender, ParperlessBilling, Partner, PhoneService, SeniorCitizen
##No Internet Service Features: DeviceProtection, MultipleLines, OnlineBackup, OnlineSecurity, StreamingMovies, StreamingTV,TechSupport

binary_encoding <- c("Dependents",  "PaperlessBilling", "Partner", "PhoneService", "DeviceProtection", "MultipleLines",
                     "OnlineBackup", "OnlineSecurity", "StreamingMovies", "StreamingTV", "TechSupport", "Churn")
train <- train %>%
  mutate(across(all_of(binary_encoding),
                ~factor(ifelse(. == "Yes", 1, 0))))

#"gender":
train <- train %>% mutate(gender = ifelse(gender == "Female", 1, 0))


## 7.2. One Hot Encoding ---------------------------------------------------
## features: Contract, InternetService, Payment Method

one_hot_features <- c("Contract", "InternetService", "PaymentMethod")

train <- dummy_cols(
  train,
  select_columns = one_hot_features,
  remove_first_dummy = FALSE,
  remove_selected_columns = TRUE )

library(janitor)

train <- train %>%
  clean_names()

# 8. FEATURE SELECTION -------------------------------------
## 8.1 Drop ID col  ---------------------------------------------------

train <- train %>% select(-id)


## 8.2. No variant features -----------------------------------------------
nearZeroVar(train, saveMetrics = TRUE)
## Insight: There isn't any no variant feature


## 8.3. Information Gain ---------------------------


ig <- information_gain(churn ~ ., data = train)

ig <- ig %>%
  arrange(desc(importance))

top_10 <- ig %>%
  slice_head(n = 10) %>%
  pull(attributes)

top_10



## 8.4. Random Forest as Feature Selector ----------------------------------

train$cchurn <- as.factor(train$churn)
train_top10 <- train %>%
  select(all_of(c(top_10, "churn")))

rf_top10 <- randomForest(
  churn ~ .,
  data = train_top10,
  ntree = 500,
  importance = TRUE
)
print(rf_top10)
importance(rf_top10)
varImpPlot(rf_top10)



## 8.1 Churn Drivers Ranking  ---------------------------------------------------
churn_rate <- mean(train$Churn == "1")

ranking <- train %>%
  pivot_longer(cols = all_of(cat_features), names_to = "variable", values_to = "category") %>%
  group_by(variable, category) %>%
  summarise(churn_rate_var = mean(Churn == "Yes"), n = n(), .groups = "drop") %>%
  mutate(impact = churn_rate_var / churn_rate) %>%
  arrange(desc(impact))

ranking

####### Insights:
# Idosos dão 2.22 vezes mais churn que a média do dataset
# Quem faz pagamento manual eletrônico cancela 2.17 vezes mais
# Clientes que não tem fidelidade de contrato, ou seja, pagam mês a mês cancelam 1.87 vezes mais
# Clientes da internet de fibra ótica cancelam os serviços 1.81 vezes mais


# 7.2. Features relevance  ---------------------------------------------------
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

# 8. MODELING -------------------------------------------------------------

## 8.1. Logistics Regression -----------------------------------------------


## 8.2. Random Forest ------------------------------------------------------


## 8.3. XGBoost ------------------------------------------------------------



