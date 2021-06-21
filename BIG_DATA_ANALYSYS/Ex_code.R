##################################################################################
########################### 작업형 제 2 유형  ####################################
##################################################################################

## package 설치
# install.packages('xgboost')
# install.packages('caret')
# install.packages('foreach')
# install.packages('doParallel')
library('xgboost')
library('caret')
library('dplyr')

##################################################################################
########################### 1. 예측 모형 (Numeric) ###############################
##################################################################################

## 데이터 불러오기
X_train <- read.csv('C:/Users/Kyean/Desktop/BigDataEx/Dataset/X_train.csv')
Y_train <- read.csv('C:/Users/Kyean/Desktop/BigDataEx/Dataset/Y_train.csv')
X_test <- read.csv('C:/Users/Kyean/Desktop/BigDataEx/Dataset/X_test.csv')

## 주구매상품 재범주화
food <- c('가공식품', '건강식품', '농산물', '수산품', '육류', '젓갈/반찬', '주류', '축산가공', '차/커피') 
ind_food <- which(X_train$주구매상품 %in% food)
need <- c('가구', '대형가전', '소형가전', '주방가전', '침구/수예', '통신/컴퓨터', '생활잡화', 
          '섬유잡화', '시티웨어', '식기', '아동', '일용잡화', '주방용품', '화장품')
ind_need <- which(X_train$주구매상품 %in% need)
hobb <- c('골프', '스포츠', '악기', '명품', '보석')
ind_hobb <- which(X_train$주구매상품 %in% hobb)
clot <- c('구두', '남성 캐주얼', '남성 트랜디', '남성정장', '디자이너', '란제리/내의', '모피/피혁',
          '셔츠', '캐주얼', '피혁잡화', '액세서리')
ind_clot <- which(X_train$주구매상품 %in% clot)
etcp <- c('기타', '커리어', '트래디셔널')
ind_etcp <- which(X_train$주구매상품 %in% etcp)

# food : 1 / need : 2 / hobb : 3 / clot : 4 / etcp : 5 로 재범주화
X_train[, 5] <- as.integer( X_train[, 5])
X_train[ind_food, 5] <- 1
X_train[ind_need, 5] <- 2
X_train[ind_hobb, 5] <- 3
X_train[ind_clot, 5] <- 4
X_train[ind_etcp, 5] <- 5
X_train[, 5] <- factor(X_train[, 5])

## 주구매지점 재범주화
bons <- c('본  점', '부산본점') 
ind_bons <- which(X_train$주구매지점 %in% bons)
seos <- c('강남점', '관악점', '노원점', '미아점', '영등포점', '잠실점', '청량리점')
ind_seos <- which(X_train$주구매지점 %in% seos)
etcs <- c(setdiff(levels(X_train[, 6]), c(bons, seos)))
ind_etcs <- which(X_train$주구매지점 %in% etcs)

# bons : 1 / seos : 2 / etcs : 3 로 재범주화
X_train[, 6] <- as.integer(X_train[, 6])
X_train[ind_bons, 6] <- 1
X_train[ind_seos, 6] <- 2
X_train[ind_etcs, 6] <- 3
X_train[, 6] <- factor(X_train[, 6])

## 주구매상품, 주구매지점 변수가 factor이므로 one-hot encoding이 필요
dummy <- dummyVars('~.', data = X_train)
X_train <- data.frame(predict(dummy, newdata = X_train))

# 환불 금액에서 nan 값 처리 방법 결정 : 0으로 대체
na_ind <- which(is.na(X_train[, 4]))
X_train[na_ind, 4] <- 0

# 환불 금액이 총구매액 혹은 최대 구매액보다 큰 경우?
# -> 이러한 경우는 없지만 환불금액만 존재하는 데이터가 존재
X_train[, 2] <- X_train[, 2] + X_train[, 4]
X_train[, 3] <- X_train[, 3] + X_train[, 4]

# cust_id 순서로 정렬
X_train <- X_train[order(X_train$cust_id), ]
Y_train <- Y_train[order(Y_train$cust_id), ]
rownames(X_train) <- X_train$cust_id
rownames(Y_train) <- Y_train$cust_id
X_train <- X_train[, -1]
Y_train <- Y_train[, -1]
X_train <- as.matrix(X_train)
Y_train <- as.matrix(Y_train)

## training model 구현
model_1 <- xgboost(data = X_train, label = Y_train, nrounds = 150, early_stopping_rounds = 100,
                   verbose = FALSE, eta = 0.05, eval_metric = 'auc')

X_test[, 5] <- as.integer(X_test[, 5])
X_test[ind_food, 5] <- 1
X_test[ind_need, 5] <- 2
X_test[ind_hobb, 5] <- 3
X_test[ind_clot, 5] <- 4
X_test[ind_etcp, 5] <- 5
X_test[, 5] <- factor(X_test[, 5])
na_ind <- which(is.na(X_test[, 4]))
X_test[na_ind, 4] <- 0

# bons : 1 / seos : 2 / etcs : 3 로 재범주화
X_test[, 6] <- as.integer( X_test[, 6])
X_test[ind_bons, 6] <- 1
X_test[ind_seos, 6] <- 2
X_test[ind_etcs, 6] <- 3
X_test[, 6] <- factor(X_test[, 6])
dummy <- dummyVars('~.', data = X_test)
X_test <- data.frame(predict(dummy, newdata = X_test))
X_test <- as.matrix(X_test[, -1])

model_1
# predict(model_1, X_test)
# write.csv(predict(model_1, X_test), file = 'C:/Users/Kyean/Desktop/20210618.csv')

##################################################################################

model_1

## 변수선택
# 변수 중요도 확인
imp <- xgb.importance(model = model_1)

# 다중공선성 확인


## parameter 튜님
grid <- expand.grid(eta = seq(0.1, 0.4, 0.05), gamma = seq(0, 5, 1))
class(grid)
head(grid)

# 병렬처리
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

grid_search <- foreach(i = 1:nrow(grid), .combine = rbind, .packages = c('dplyr', 'xgboost')) %dopar% {
  model = xgb.cv(data = X_train, label = Y_train, nrounds = 200, nfold = 5,
                 early_stopping_rounds = 150, verbose = FALSE, 
                 prediction = TRUE, params = grid[i, ])
  data.frame(train_rmse_last = unlist(model$evaluation_log[, 2]) %>% last, 
             test_rmse_last = unlist(model$evaluation_log[, 4]) %>% last)
}

##################################################################################
########################### 2. 분류 모형 (Binary) ################################
##################################################################################

# 전처리
# 변수 선택
# 모형 : random forest | XGboost package 사용
# Parameter 최적화
# 앙상블


##################################################################################
########################### 3. 분류 모형 (Multiple) ##############################
##################################################################################

# 전처리
# 변수 선택
# 모형 : random forest | XGboost package 사용
# Parameter 최적화
# 앙상블
