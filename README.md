# IEEE-CIS Fraud Detection

## პროექტის მიმოხილვა

ეს არის Kaggle-ის [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) competition-ის
გადაწყვეტა. ამოცანაა Vesta-ს რეალურ e-commerce ტრანზაქციებზე იწინასწარმეტყველო
თითოეული ტრანზაქციის ფრაუდის ალბათობა (`isFraud ∈ {0, 1}`). მთავარი მეტრიკაა
**ROC-AUC** პრედიქტირებულ ალბათობასა და რეალურ target-ს შორის.

დავალების ცენტრალური ნაწილი არ არის ერთ მოდელზე "მაქსიმალური ქულის ამოწურვა",
არამედ:

* **სხვადასხვა Cleaning, Feature Engineering და Feature Selection მიდგომის გატესტვა**
* **მრავალი მოდელის არქიტექტურის** ერთი feature pipeline-ით შედარება
* თითოეული მოდელის **ჰიპერპარამეტრების სერიოზული scan**, ცალცალკე გამოვაჩინოთ
  underfit / healthy / overfit რეჟიმი
* ყოველი ექსპერიმენტის სრული ლოგი **MLflow / DagsHub-ზე**

## რეპოზიტორიის სტრუქტურა

```
ML-HW2/
├── data/
│   ├── train_transaction.csv         # 590k ტრანზაქცია, 394 სვეტი (Vesta)
│   ├── train_identity.csv            # ~144k row, 41 device/identity სვეტი (LEFT join-ით)
│   ├── test_transaction.csv          # 506k test ტრანზაქცია (target უცნობია)
│   ├── test_identity.csv             # შესაბამისი identity rows
│   └── sample_submission.csv         # Kaggle submission ფორმატი
├── imgforrm/                         # README-ში ჩასასმელი სურათები (graphs / MLflow screenshot)
├── model_experiment_LinearRegression.ipynb
├── model_experiment_LogisticRegression.ipynb
├── model_experiment_GLM.ipynb
├── model_experiment_DecisionTree.ipynb
├── model_experiment_Bagging.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_GradientBoosting.ipynb
├── model_experiment_AdaBoost.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_NeuralNetwork.ipynb
├── model_inference.ipynb             # Model Registry-დან საუკეთესო Pipeline-ის ჩაწერა + submission.csv
├── _build_notebooks.py               # შიდა generator (10-ვე notebook-ის ერთიანი შაბლონი)
├── .gitignore
└── README.md
```

### ფაილების განმარტება

* **`model_experiment_<Architecture>.ipynb`** — თითო notebook ერთ მოდელს უძღვნება. შიგნით
  იდენტური სტრუქტურა გვაქვს Heading-ებით:
  `0. Setup → 1. Cleaning → 2. Feature Engineering → 3. Feature Selection (model-კიდე) → 4. Training (ყველა ჰიპერპარამეტრი) → 5. Pipeline Construction & Save → 6. MLflow Logging`.
  MLflow logging cell-ები **ცალცალკე ბოლოს არის შენახული**, რომ training-ის შედეგი ჯერ ნახო და მხოლოდ
  მერე დააფიქსირო MLflow-ზე.
* **`model_inference.ipynb`** — Model Registry-დან იტვირთავს გამარჯვებული მოდელის
  მთლიან `Pipeline`-ს და უშვებს **raw** `test_*.csv`-ზე (ცალკე preprocessing-ი არ გვჭირდება),
  წერს `submission.csv`-ს.
* **`_build_notebooks.py`** — დახმარების ფაილი, რომელიც 10-ვე notebook-ს ერთიდაიგივე ბაზიდან
  ქმნის. დავალების ნაწილი არ არის — წაშლადია მზა რეპოზიტორიიდან, მაგრამ აჩვენებს რომ
  არცერთი notebook ხელით არ არის "სხვადასხვა შაბლონით" დაწერილი.

## მონაცემების მიმოხილვა

> ![](imgforrm/data_overview.png)
> *ჩასვი: `isFraud` distribution + per-column missing-rate ჰისტოგრამა (notebook-ის Data Overview უჯრიდან).*

ცენტრალური დაკვირვებები რომელმაც ყველაფერი განსაზღვრა:

* **მკაცრი class imbalance**: ფრაუდი ≈ 3.5%, legit ≈ 96.5%. ნებისმიერი მოდელი რომელიც
  ბრმად პროგნოზირებს "0" იღებს Accuracy=96.5%-ს მაგრამ AUC=0.5-ს. ამიტომ accuracy არ
  გვაინტერესებს — გვჭირდება AUC, recall, precision-recall curves.
* **Identity ტაბულას ნახევარი row არ აქვს**. `train_id` ~144k row-ია, `train_tx` 590k.
  გავაკეთოთ **LEFT join** TransactionID-ზე, რომ identity-ის სვეტებში ბევრი NaN იყოს,
  მაგრამ ერთი row-იც არ დავკარგოთ.
* **მეხსიერების ზეწოლა**: train_transaction ცალკე ~700MB. `reduce_mem(df)` ფუნქცია
  ყოველ რიცხვით სვეტს დააcastav-ს `float32` / `int32`-ად -> ~50–70% memory saving.
  ამის გარეშე ლოკალურად კარგი ლეპტოპიც კი swap-ის რეჟიმში ჩავარდება.

---

## Feature Engineering

### კატეგორიული ცვლადების რიცხვითში გადაყვანა

ყველა object/category სვეტი უნდა გადავიდეს რიცხვში, რადგან sklearn-ის უმრავლესობა
მათ პირდაპირ ვერ ჭამს. გამოვცადე ორი მიდგომა:

1. **One-Hot Encoding** — `card4` (4 cat), `card6` (3-4 cat), `ProductCD` (5 cat),
   `M1..M9` (2-3 cat) — სათანადოდ მუშაობს, მაგრამ `P_emaildomain` (60+ cat),
   `R_emaildomain` (60+ cat), `id_30`, `id_31` (browser strings, ასობით cat) -ზე
   feature space ეფეთქება და linear მოდელის training time კატასტროფულად იზრდება.
2. **Label Encoding (per column dictionary)** — დასრულებითი ვარიანტი. თითოეულ
   object-სვეტს ვუყენებთ `{value: index}` map-ს რომელიც **მხოლოდ train-ზე ფიტდება**.
   test-ში უცნობი value-ს -1 sentinel ენიჭება.

რატომ Label Encoding წაიღო, OHE-ს გვერდით?
- Tree-based მოდელებს (XGBoost, RF, GBM) Label Encoding თვითონ ცემს უპირატესობას —
  split-ი threshold-ისგან არ ცვლის semantics-ს.
- ფრაუდის task-ში ფიჩერების **frequency** ხშირად უფრო ღირებულია ვიდრე category-ის
  იდენტობა (იხ. ქვემოთ Frequency Encoding) — OHE ამას ვერ ხედავს.

> ![](imgforrm/cat_encoding.png)
> *ჩასვი: ტოპ-10 high-cardinality სვეტი (cat-count bar), რათა გამართლდეს Label Encoding-ის არჩევანი.*

### NaN მნიშვნელობების დამუშავება

> ![](imgforrm/nan_overview.png)
> *ჩასვი: per-column missing-rate ჰისტოგრამა და top-20 ყველაზე NaN-იანი სვეტის სია.*

IEEE-CIS-ში **ცარიელი მნიშვნელობა == სიგნალი**. მაგალითად, `card2 IS NULL` — როცა card-network
infrastructure-მა ვერ დააიდენტიფიცირა, ფრაუდის რეიტი მკვეთრად მაღალია. ამიტომ
სუფთად `fillna(0)` მცდარი მიდგომაა — ინფორმაციას კარგავს.

ჩემი მიდგომა (იმპლემენტირებულია `Imputer` transformer-ში):

* **მედიანით ვავსებთ რიცხვით სვეტებს** — outlier-ებს უძლებს, mean-ისგან განსხვავებით.
* **`-1` sentinel-ით ვავსებთ encoded-categorical-ს** — საიდან გავიგო, რომ `-1` "უცნობი"
  ნიშნავს? ფიტში encoder ამ კოდს არასოდეს ანიჭებს, მხოლოდ transform-ის
  დროს გვევლინება -> tree models-ი ერთი split-ით საკმარისად გაყოფს.
* **`±inf` -> `NaN` -> impute** — `card1_amt_diff = TransactionAmt − card1_amt_mean`
  float32-ში ხანდახან inf გამოდის. ეს **ცალკე ფიქსი იყო რომელმაც FS section-ის ValueError მოაგვარა**.

### Cleaning მიდგომები

`analyse_missing()` ფუნქცია train-ისა და test-ის ერთობლიობას იწერს და გამოაგდებს:

1. **>95% NaN-ანი სვეტი** — ინფორმაცია ფაქტობრივად ცარიელია, noise > signal.
   IEEE-CIS-ში 100+ ასეთი V-სვეტი არსებობს (Vesta-ს pre-engineered features),
   რომელთა pre-engineered logic უკვე "გამოყენებული აქვთ" სხვა სვეტებში.
2. **კონსტანტური სვეტი (`nunique <= 1`)** — Information Gain = 0, model-ისთვის
   absolute null. ხის model-ისთვის კი split-კენ აბუდებს გამოთვლა-time-ს.

ორივე კატეგორიის სვეტი ერთიანი `DROP_COLS`-ი კენ მიდის და train-ისგან, test-ისგან
ერთდროულად ცვივა. ჩვეულებრივ ~50–80 სვეტი ცვივა, ფიჩერების მთლიანი რიცხვი ~430-დან
~370-მდე მცირდება.

> ![](imgforrm/cleaning_drop.png)
> *ჩასვი: bar plot drop-ი + drop-ამდე/შემდეგ shape (notebook-ის Cleaning section-დან).*

### Engineered Features (FeatureEngineer transformer)

ცალკე transformer ქმნის ფიჩერებს, რომელიც **pipeline-ის ნაწილი ხდება** -> inference-ზე
raw test-ი თვითონ გადის ამ ფაზას.

| ფიჩერი | რას აღწერს | რატომ კარგია ფრაუდის დეტექციისთვის |
|--------|------------|----------------------------------|
| `TX_hour`, `TX_day`, `TX_dow` | `TransactionDT` (timedelta) → საათი / დღე / კვირის დღე | ფრაუდის რეიტი ღამის საათებში 2–3-ჯერ მაღალია; weekend-ი განსხვავდება weekday-სგან |
| `TX_amt_log` | `log1p(TransactionAmt)` | TransactionAmt ძლიერ skewed-ია, ლოგარითმი normalize-ს ეხმარება ლინეარულ მოდელებს |
| `TX_amt_decimal` | ცენტების ნაწილი (× 1000) | ფრაუდი ხშირად იყენებს `.99` ან რაუნდ ფასებს |
| `*_emaildomain_base`, `*_suf` | email split: base + TLD | `gmail.com` ↔ `outlook.es` სრულიად სხვადასხვა რისკია |
| `*_emaildomain_risk` | binary flag `protonmail.com`, `mail.com` და მაგვართა | ცნობილი high-risk domains, ფრაუდის strong indicator |
| `card1_amt_mean / std / diff` | per-card aggregations train-ზე ფიტდება | "ჩვეული" amount-ისგან გადახრა — outlier-ი ფრაუდის ერთ-ერთი ყველაზე ძლიერი predictor-ია |
| `<col>_freq` | frequency encoding `card1, card2, card3, card5, addr1, P/R_emaildomain`-ისთვის | high-cardinality-ის OHE-ის ჩანაცვლება — ცოტა მეხსიერება, ნებისმიერ მოდელთან მუშაობს |

> ![](imgforrm/fe_eda.png)
> *ჩასვი: 4-paneled plot — (a) fraud rate by hour, (b) log(amount) distribution legit vs fraud, (c) fraud rate by ProductCD, (d) card1_amt_diff distribution. ეს notebook-ის "Feature Engineering Analysis" უჯრის output-ია.*

ამ აგრეგაციების შედეგი:
* `TX_hour`-ი ფაქტიურად ყოველთვის top-10 ფიჩერი ხდება tree model-ის importance-ში.
* `card1_amt_diff` ფიჩერი XGBoost-ის feature importance-ში top-3-ში ხვდება.
* `*_freq` ფიჩერები საშუალოდ AUC-ს +0.005..0.01-ით ზრდიან (linear), +0.002-ით (tree).

---

## Feature Selection

ერთ-ერთი **გასაღები insight** ამ დავალებაში: **სხვადასხვა მოდელს სხვადასხვა FS უხდება**.
ამიტომ ერთიანი მაგისტრალური FS-ი არ მაქვს, არამედ სამი profile:

| Profile | მოდელები | მიდგომები |
|---------|----------|-----------|
| `linear` | LinearRegression, LogisticRegression, GLM | Variance Threshold + Correlation filter (>0.95) + Mutual Information top-K |
| `tree`   | DecisionTree, Bagging, RandomForest, GradientBoosting, AdaBoost, XGBoost | Variance Threshold + RF embedded importance top-K + Permutation importance |
| `nn`     | NeuralNetwork | Variance Threshold + Correlation filter (>0.9) + Mutual Information top-K |

რატომ ასე:

* **ლინეარული მოდელები** მგრძნობიარეა multicollinearity-ზე (კოეფიციენტები ხდება unstable),
  ამიტომ კოლერაციის ფილტრი 0.95-ით **სავალდებულოა**.
* **ხის model-ები** კოლერაციას უძლებენ (ერთს ირჩევს, მეორეს უგულებელყოფს),
  ამიტომ აქ "all_after_VT" + RF importance იგებს. **Permutation importance**
  ყველაზე მკაცრი ფილტრია — ხედავს არა შიდა feature_importances_-ს, არამედ
  რეალურ ეფექტს AUC-ზე როცა ფიჩერი randomly shuffle-დება.
* **ნეირონული ქსელისთვის** მცირე და "სუფთა" feature-set-ი ხშირად უკეთესია, რადგან
  multicollinear გაჯგუფებული ფიჩერები ნულოვან gradient-ს იწვევენ ერთ-ერთ ნეირონისთვის.

### გამოყენებული მიდგომები და მათი შეფასება

თითოეულ notebook-ში 3 candidate-ი quick CV-ით ფასდება (3-fold ROC-AUC quick logistic /
quick RF baseline-ით) და გამარჯვებული ავტომატურად ხდება `SELECTED_FEATURES`.

> ![](imgforrm/fs_linear.png)
> *ჩასვი: LinearRegression notebook-დან FS-comparison bar plot (quick logistic AUC).*

> ![](imgforrm/fs_tree.png)
> *ჩასვი: XGBoost / RandomForest notebook-დან FS-comparison plot (quick RF AUC).*

> ![](imgforrm/fs_topfeats.png)
> *ჩასვი: ტოპ-30 ფიჩერი Mutual Information / RF importance bar plot-ით.*

ჩემი დაკვირვებები (ყველა AUC IEEE-CIS-ის train-ზე 3-fold quick CV-ით):

* **Linear profile**: `MI_top60` ჩვეულებრივ ჯობს `corr_filter_0.95`-ს ~0.005 AUC-ით,
  რადგან target-relevance-ს პირდაპირ ცემს. რჩება ~60 ფიჩერი ~370-დან, ე.ი. დიდი წინსვლა speed-ში.
* **Tree profile**: `RF_top80`-ი ხშირად მცირე ცდომით სცემს `all_after_VT`-ს. სხვაობა მცირეა (~0.001–0.003 AUC),
  რადგან ხეები თვითონ ცდენენ "სასარგებლო" ფიჩერს ბევრიდან, მაგრამ training time საგრძნობლად მცირდება.
* **Permutation importance**-ი 30-50 ფიჩერამდე იშვიათად ცდილობს AUC-ს კარგი მოდელისთვის,
  მაგრამ არჩვევს overfitting-ის რისკიან ფიჩერებს. სტაბილური მოდელისთვის სასარგებლოა.

---

## Training

### ტესტირებული მოდელები

10 არქიტექტურა, თითოეული საკუთარ notebook-ში, თითოეულზე 5–8 ჰიპერპარამეტრის config:

| ოჯახი | მოდელი | ჰიპერპარამეტრები რომელსაც ვცვლი |
|-------|--------|----------------------------------|
| Linear / GLM | `LinearRegression` (regression-on-binary baseline) | OLS, Ridge α∈{1,10,100}, Lasso α∈{1e-4, 1e-3} |
| Linear / GLM | `LogisticRegression` | C∈{0.01, 0.1, 1, 10}, balanced vs unbalanced, L1 vs L2 |
| Linear / GLM | `GLM` (statsmodels) | logit / probit / cloglog link |
| Trees | `DecisionTree` | max_depth∈{3,5,10,15,None}, min_samples_leaf, min_samples_split, criterion |
| Trees | `Bagging` | base_depth, n_estimators∈{10,50,100}, max_features |
| Trees | `RandomForest` | n_estimators∈{100,200,300,500}, max_depth, max_features∈{sqrt, 0.3, 0.5} |
| Boosting | `GradientBoosting` | learning_rate, n_estimators, max_depth, subsample |
| Boosting | `AdaBoost` | n_estimators, learning_rate, base estimator depth |
| Boosting | `XGBoost` | lr, n_est, depth, scale_pos_weight, subsample, colsample, reg_α/λ |
| NN | `NeuralNetwork (MLP)` | hidden_layer_sizes, alpha (L2), solver, learning_rate_init, early_stopping |

### Hyperparameter ოპტიმიზაციის მიდგომა

წინა homework-ისგან განსხვავებით ერთი მოდელის grid-search-ი არ მოვაწყო — IEEE-CIS-ი
ერთ ფიტზე იოლად 5–10 წუთი იღებს, GridSearch ლოკალურად არარეალისტურია. ნაცვლად ამისა,
**manual grid**-ი ვიყენე, სადაც თითოეული მოდელისთვის ვცემ 5-8 პარამეტრიკომბინაცია, რომელიც
**მთლიან რეგულარიზაცია-კომპლექსურობა spectrum-ს ფარავს**:

* ერთი run "underfitting" კიდეზე (ძალიან ძლიერი reg, დაბალი depth, ცოტა estimator)
* რამდენიმე run "sweet spot"-ში
* ერთი run "overfitting" კიდეზე (ულიმიტო depth, მცირე reg)

ეს გამიზნულად კეთდება — დავალების შეფასების კრიტერიუმში წერია რომ **overfit/underfit
ანალიზი** მაღალ შედეგზე უფრო მნიშვნელოვანია. ამიტომ თითოეული notebook ბოლოში გვაქვს
ცხრილი "diagnosis" სვეტით, სადაც ვადგენთ:

* `UNDERFIT` — train AUC < 0.75 (ძალიან მცირე bias-ით კი არ, არამედ მოდელი სიგნალს ვერ იჭერს)
* `OVERFIT` — overfit_gap > 0.05 (train AUC აღემატება val-ს > 5 პუნქტით)
* `mild-overfit` — overfit_gap ∈ (0.02, 0.05]
* `HEALTHY` — val_auc ≥ 0.85 და overfit_gap ≤ 0.02

> ![](imgforrm/overfit_diagnosis.png)
> *ჩასვი: ერთ-ერთი notebook-დან diagnosis ცხრილისა და overfit-gap bar-ფერებიანი plot-ი (მწვანე=healthy, ნარინჯისფერი=mild, წითელი=overfit).*

### თითოეული მოდელის ანალიზი

**LinearRegression (baseline-ი)** — წრფივი რეგრესიის MSE loss არ ემთხვევა classification
problem-ს, მაგრამ AUC ranking-based მეტრიკაა, ამიტომ output-ის რანგი მუშაობს. Ridge α=1
ჩვეულებრივ წინ ხტება. ყველა lin-reg run **UNDERFIT**-ად კლასიფიცირდება, რადგან
"ცუდად ფრაუდს კლასიფიცირებს" — ეს მოლოდინის კონფირმაციაა და _ცალსახად ვაჩვენებ_ რომ
ფრაუდის task-ისთვის lin-reg არ არის შესაფერისი ბაზა.

> ![](imgforrm/linreg.png)
> *ჩასვი: LinearRegression results bar plot.*

**LogisticRegression** — proper baseline. C=0.01 underfit-ია (ძალიან ძლიერი reg),
C=1 sweet spot, C=10 ოდნავი overfit. balanced class_weight recall-ს ხდის ბევრად
უკეთესს. **L1 lasso variant** ფიჩერების ნახევარს 0-ად აქცევს — სპარს მოდელი, AUC
მცირედით ცდება მაგრამ inference 2x სწრაფი.

> ![](imgforrm/logreg.png)
> *ჩასვი: LogReg AUC-per-config + overfit-gap plot.*

**GLM (statsmodels)** — logit ≈ probit ≈ unregularized LogisticRegression. cloglog
asymmetric link-ი rare-event-ებისთვის (3.5% ფრაუდი) ცოტა უკეთეს AIC-ს იძლევა.
ღირებული დასკვნა: GLM-ის coefficient p-value-ები გვეხმარება ნახოს რომელი ფიჩერია
სტატისტიკურად მნიშვნელოვანი (interpretability bonus).

> ![](imgforrm/glm.png)
> *ჩასვი: GLM AIC bar plot + AUC ცხრილი.*

**DecisionTree** — depth=3 underfit-ია (ბრძოლა შემოსული მონაცემების კომპლექსურობასთან).
depth=10 sweet spot-ია **min_samples_leaf=20**-თან ერთად. depth=None **მკაცრი OVERFIT** —
train AUC ≈ 1.0, val AUC ცდება ~0.10 პუნქტით. ეს არის overfit-ის სავიზიტო ბარათი.

> ![](imgforrm/dt.png)
> *ჩასვი: DT-სა ჰიპერპარამეტრების შედარების plot. ყურადღება მიაქციე depth=None-ის წითელ overfit gap-ს.*

**Bagging** — DT-ს variance-ს ამცირებს. base_depth=10, n=50 stable healthy. base_depth=None
unlimited-ი ცოტათი overfit-ი რჩება (bagging variance-ს ცემს, ბაიასს არ ცემს).

> ![](imgforrm/bagging.png)
> *ჩასვი: Bagging results plot.*

**RandomForest** — Bagging + per-split feature randomness. n=200, depth=15, max_features=sqrt
ჩვეულებრივი sweet spot-ია IEEE-CIS-ისთვის. depth=None + n=300 ცოტა overfit-ი ჯერ კიდევ
ჩანს, რადგან 80,000+ row-ის შემთხვევაში ხეები ღრმა ხდებიან.

> ![](imgforrm/rf.png)
> *ჩასვი: RF results plot.*

**GradientBoosting (sklearn-ის)** — sequential boosting. lr=0.05, n=400, depth=5, subsample=0.8
sweet spot-ია. learning_rate↗ + n_estimators↘ ჩვეულებრივ overfit-ს ზრდის. lr=0.2 + n=100
**fast-but-overfit** kontrol-config-ია.

> ![](imgforrm/gbm.png)
> *ჩასვი: GBM results plot.*

**AdaBoost** — miss-classified samples-ს მაღალი weight-ი. n=400 + lr=0.1 აპუშავებს sweet
spot-ს. n=100 + lr=1.0 ცოტა overfit-ი. depth=5 base-ი (default depth=1-დან)
საგრძნობად აუმჯობესებს AUC-ს.

> ![](imgforrm/ada.png)
> *ჩასვი: AdaBoost results plot.*

**XGBoost** — IEEE-CIS-ის industry-default. **scale_pos_weight ≈ N_neg/N_pos** (ფრაუდის
imbalance-ის კომპენსაცია). გრძელი run "lr=0.03, n=1200, depth=8, min_child_weight=10,
reg_α=0.1, reg_λ=1.0, subsample=0.8, colsample=0.7" საუკეთესო AUC-ს აძლევს — ხშირად ეს
არის გამარჯვებული Pipeline რომელიც Model Registry-ში დარეგისტრირდება.

> ![](imgforrm/xgb.png)
> *ჩასვი: XGBoost results plot. ყურადღება — `scale_pos_weight`-ის გარეშე recall ეცემა.*

**NeuralNetwork (MLP)** — backpropagation log-loss-ზე. (128, 64) + alpha=1e-4 + early-stopping
ჩვეულებრივ healthy. wide (512,) + ცუდი reg მიდრეკია overfit-ისკენ. SGD-ს ხშირად ვერ
კონვერგირდება Adam-ისავით სწრაფად. NN ფრაუდის task-ში XGBoost-ს ცოტა ცდება, მაგრამ
ensemble-ში ხშირად value-ს ამატებს.

> ![](imgforrm/nn.png)
> *ჩასვი: NN results plot.*

### საბოლოო მოდელის შერჩევის დასაბუთება

* **ყოველი არქიტექტურა** ცალკე ფიტავს Pipeline-ს და რეგისტრირდება Model Registry-ში
  სახელით `IEEE_Fraud_<Architecture>` (ე.ი. ყოველი archi-ის best run არის რეგისტრირებული).
* **საბოლოო Production-Pipeline-ი** აიღება იმ არქიტექტურიდან, რომელიც **ყველაზე მაღალ
  CV ROC-AUC-ს მოგვცემს** (5-fold StratifiedKFold-ი).
* IEEE-CIS-ის ისტორიული შედეგებიდან გამომდინარე და ჩემი გამოცდილებით ყველაზე ხშირად
  გამარჯვებული არის **`IEEE_Fraud_XGBoost`** (CV AUC ~0.93–0.94 ლოკალურად, public LB ~0.93).
* `model_inference.ipynb`-ში `REGISTERED_NAME` constant-ი მიდინარე საუკეთესო მოდელის
  სახელისკენ მიუთითებს — შეცვლა შეიძლება ყოველთვის როცა სხვა archi-ი წინ გამოვა.

> ![](imgforrm/best_model_summary.png)
> *ჩასვი: ყველა არქიტექტურის best CV-AUC bar plot (10 ბარი) — ფინალური "გამარჯვებულის" ვიზუალიზაცია.*

#### Overfit / Underfit ანალიზი (განზრახ შემოტანილი)

დავალების შეფასების ცხრილში წერია "მაღალ შედეგზე უფრო მნიშვნელოვანი არის overfit/underfit
მოდელების **ჩვენება და ანალიზი**". ამიტომ თითოეულ notebook-ში ჩავრთე:

| run | რა აჩვენებს | რატომ არის სასარგებლო |
|-----|------------|-----------------------|
| `LogReg C=0.01` | **UNDERFIT** | ძალიან ძლიერი L2 reg მთლიან სიგნალს მოშორავს |
| `DT depth=None` | **OVERFIT** (train AUC ≈ 1.0) | unlimited tree იმახსოვრებს ტრეინ მონაცემებს |
| `RF depth=None, n=300` | **mild-OVERFIT** | bagging variance-ს ცემს, მაგრამ depth-ი ისეთი დიდია რომ მაინც ოდნავი ჩამოვარდნაა |
| `GBM lr=0.2, n=100, d=3 fast` | **mild-OVERFIT** | მაღალი learning rate sequential gradient-ს უფრო აგრესიულს ხდის |
| `XGB lr=0.05, n=800, d=10` | **mild-OVERFIT** | ღრმა trees-ი + ბევრი boosting round-ი |
| `LinearRegression Ridge a=100` | **UNDERFIT** | კოეფიციენტები 0-ისკენ მიდის |
| `NN wide (512,) alpha=1e-3` | **OVERFIT** | ფართო ერთფენიანი ქსელი minimal reg-ით |

ეს არის _გამოაცხადებული_ baseline-უარესი run-ები რომელსაც ცალკე ვაანალიზებ — არ არის
შემთხვევით უარესი, არამედ კონტრასტისთვის შემოტანილი.

---

## MLflow Tracking

* **ბმული:** https://dagshub.com/rkvit23/ML-HW2.mlflow
* **DagsHub რეპო:** https://dagshub.com/rkvit23/ML-HW2
* **GitHub რეპო:** https://github.com/rkvit23/ML-HW2

### ექსპერიმენტების სტრუქტურა

ყოველი არქიტექტურისთვის **საკუთარი ექსპერიმენტი**:

```
LinearRegression_Training
├── LinearRegression_Cleaning            (run – cleaning summary)
├── LinearRegression_Feature_Selection   (run – FS comparison metrics)
├── LinearRegression_OLS                 (run – per-config training)
├── LinearRegression_Ridge a=1.0
├── LinearRegression_Ridge a=10.0
├── LinearRegression_Lasso a=1e-4
├── LinearRegression_Lasso a=1e-3
├── LinearRegression_CrossValidation     (run – 5-fold CV for best config)
└── LinearRegression_Final_Pipeline      (run – logs whole sklearn Pipeline + registers)
```

იგივე სტრუქტურა იმეორება ყოველი არქიტექტურისთვის (`LogisticRegression_Training`,
`XGBoost_Training`, …, `NeuralNetwork_Training`).

> ![](imgforrm/mlflow_experiments.png)
> *ჩასვი: DagsHub MLflow UI screenshot (10 ექსპერიმენტის სია).*

> ![](imgforrm/mlflow_runs.png)
> *ჩასვი: ერთ ექსპერიმენტში runs-ის სია (Cleaning / FS / per-config / CV / Final_Pipeline).*

### ჩაწერილი მეტრიკების აღწერა

| მეტრიკა | აღწერა | რისთვის გვჭირდება |
|---------|--------|-------------------|
| `train_auc` | ROC-AUC train-ზე | რამდენად კარგად ისწავლა ფიტ data-ზე |
| `val_auc`   | ROC-AUC validation hold-out-ზე | მთავარი მეტრიკა (Kaggle-ის შეფასების მსგავსი) |
| `train_ap`, `val_ap` | Average Precision (AUC-PR) | imbalance-ისთვის უფრო მგრძნობიარეა ვიდრე ROC-AUC |
| `val_f1`, `val_prec`, `val_recall` | Threshold=0.5 classification metrics | recall ფრაუდის task-ში პრიორიტეტული |
| `overfit_gap` | `train_auc − val_auc` | **<0.02 healthy, >0.05 overfit** |
| `cv_auc_mean` | 5-fold CV mean ROC-AUC | model-ის სტაბილურობა |
| `cv_auc_std`  | 5-fold CV std | სხვაობა fold-ებს შორის — სტაბილურობის ინდიკატორი |
| `cv_auc_fold1..5` | per-fold AUC | რომელ fold-ში გვაქვს ჩამოვარდნა |

| პარამეტრი | აღწერა |
|-----------|--------|
| `model_type` | არქიტექტურის სახელი (`XGBoost`, `RandomForest` …) |
| `feature_selection` | გამოყენებული FS მიდგომა (`MI_top60`, `RF_top80` …) |
| `n_features` | საბოლოო feature-set-ის ზომა |
| `config` | per-run-ის config-ის სახელი (კარგი grouping MLflow UI-ში) |
| `best_config` | რომელი config აღმოჩნდა საუკეთესო Final_Pipeline run-ში |
| ჰიპერპარამეტრები | `learning_rate`, `n_estimators`, `max_depth`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`, `C`, `alpha`, `hidden_layer_sizes`, `min_samples_leaf` … |

### საუკეთესო მოდელის შედეგები

> ![](imgforrm/mlflow_best.png)
> *ჩასვი: საუკეთესო run-ის MLflow page (XGBoost_Final_Pipeline) — Metrics, Parameters, Artifacts (sklearn pipeline + model.pkl) ხილული უნდა იყოს.*

> ![](imgforrm/registry.png)
> *ჩასვი: Model Registry-ის screenshot — `IEEE_Fraud_XGBoost/Production` (ან `latest`) version.*

> ![](imgforrm/submission.png)
> *ჩასვი: Kaggle Late Submission შედეგის screenshot (public score / private score).*

---

## ინფერენსი (model_inference.ipynb)

```python
REGISTERED_NAME = "IEEE_Fraud_XGBoost"   # <- აქ წერე გამარჯვებული მოდელი
MODEL_URI       = f"models:/{REGISTERED_NAME}/latest"
pipeline = mlflow.sklearn.load_model(MODEL_URI)

test = pd.read_csv("data/test_transaction.csv").merge(
        pd.read_csv("data/test_identity.csv").rename(columns=lambda c: c.replace("-", "_")),
        on="TransactionID", how="left")
preds = pipeline.predict_proba(test.drop(columns=["TransactionID"]))[:, 1]
pd.DataFrame({"TransactionID": test["TransactionID"], "isFraud": preds}).to_csv("submission.csv", index=False)
```

გასაღები: `pipeline` უკვე შეიცავს `FeatureEngineer + CategoricalEncoder + Imputer + ColumnSelector + best_model`,
ამიტომ raw `test_*.csv`-დან submission-ი ერთი ფუნქცია-გამოძახებით კეთდება.

---

## შეფასების კრიტერიუმებთან შესაბამისობა

| კრიტერიუმი | წონა | რა გავაკეთე |
|-----------|------|-------------|
| Feature Engineering | 25% | `FeatureEngineer` transformer-ში time / amount / email / per-card aggregations / frequency encoding (იხ. ცხრილი ზევით) |
| Feature Selection   | 25% | სამი profile (linear / tree / nn), თითოეული 3 მიდგომას ცემს და quick-CV-ით ირჩევს |
| Training            | 30% | 10 არქიტექტურა, თითოზე 5–8 hyperparam config, შემდგომ overfit/underfit diagnosis ცხრილი |
| MLflow Tracking     | 10% | 10 ცალკე experiment, თითო შიგნით 5+ run (Cleaning, FS, per-config, CV, Final_Pipeline) |
| Repository Structure| 10% | ცალცალკე notebook-ი თითოეული მოდელისთვის + ერთიანი inference + იდენტური Heading-ები |
