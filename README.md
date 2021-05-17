[![author](https://img.shields.io/badge/author-ghermsen-red.svg)](https://www.linkedin.com/in/gabrielhermsen/)
[![](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.9/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ghermsen/airbnb_munchen/issues)

# Rossmann Stores Sale Prediction

Dirk Rossmann GmbH, commonly known as Rossmann, is the largest drugstore chain in Germany in the number of stores, with 2196 stores in German territory (2020, [Statista](https://www.statista.com/statistics/505614/number-of-drugstore-branches-germany/#:~:text=The%20German%20drugstore%20market%20is,Budni%2C%20was%20founded%20in%20Hamburg)). In 2015, Rossmann launched a competition on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/overview) to forecast sales in stores for the next six weeks.

This project will cover all the steps of a Data Science project from understanding the problem until the deployment of a solution as seen above, and it will predict the sale of a particular store in the next six weeks.

<br><center><img alt="" width="100%" src="https://upload.wikimedia.org/wikipedia/commons/9/95/Rossmann_Schriftzug_mit_Centaur.jpg"></center>
Photo by [Anakin81](https://commons.wikimedia.org/wiki/File:Rossmann_Schriftzug_mit_Centaur.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons

Esse projeto segue a mesma proposta sujerida na competição do [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/overview), With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

---

## Table of Contents
- [Project Methodology](#project-methodology)
- [01. The Problem and the Solution](#01-the-problem-and-the-solution)
- [02. Data Understanding and Data Preparation](#02-data-understanding-and-data-preparation)
- [03. Feature Engineering](#03-feature-engineering)
- [04. Exploratory Data Analysis](04-exploratory-data-analysis)
- [05. Data Preprocessing](#05-data-preprocessing)
- [06. Feature Selection](#06-feature-selection)
- [07. Machine Learning Modeling](#07-machine-learning-modeling)
- [08. Hyperparameter Tuning](#08-hyperparameter-tuning)
- [09. Understanding the Error and Business Performance](#09-understanding-the-error-and-business-performance)
- [10. Final Model Deployment](#10-final-model-deployment)
- [11. The Final Solution](#11-the-final-solution)
- [Conclusion](#conclusion)

---

## Project Methodology
[(next section)](#the-problem-and-the-solution) | [Table of Contents](#table-of-contents)

The methodology used in this project is the **CRISP-DM** (Cross-Industry Standard Process for Data Mining), one of the most used methodologies in data science projects. In my opinion, the main advantages of using this methodology in data science projects (when compared to others SEMMA and KDD) are that any industry can use CRISP-DM on its projects and the agility in the generation of value.

For a better understanding, we will assume the following situation: we are in the modeling phase of the project, and from our experience, it can be seen that the results can be better if a specific variable were created and implemented. However, the data preparation phase has already been completed, which makes the implementation of this new variable only happen in the second cycle of CRISP-DM. This method concludes the current cycle in less time and delivers value and knowledge at the end of each phase, automatically generating a list of future implementations for future cycles.

Below there is a diagram that shows this methodology:

<br><center><img alt="" width="50%" src="https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png"></center>
Photo by [Kenneth Jensen](https://commons.wikimedia.org/wiki/File:CRISP-DM_Process_Diagram.png), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0), via Wikimedia Commons

---

## 01 - The Problem and the Solution
[(next section)](#data-understanding-and-data-preparation) | [(previous section)](#project-methodology) | [Table of Contents](#table-of-contents)

This is the first phase of the project according to the CRISP-DM. Here there is an understanding of the problem, and consequently, it is expected that there will already be the delivery of value with a suggested solution.

Analyzing the description of the competition in [Kaggle](https://www.kaggle.com/c/rossmann-store-sales), the following can be observed: * "With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied." *. With this mention, it seems that the problem lies in obtaining sales results with varying accuracy by these thousands of managers.

Thus, the project's objective is to create a machine learning model that can understand the factors that influence the sales of each store and generate forecasts following these factors.

---

## 02 - Data Understanding and Data Preparation
[(next section)](#feature-engineering) | [(previous section)](#the-problem-and-the-solution) | [Table of Contents](#table-of-contents)



---

## 03 - Feature Engineering
[(next section)](#exploratory-data-analysis) | [(previous section)](#data-understanding-and-data-preparation) | [Table of Contents](#table-of-contents)

---

## 04 - Exploratory Data Analysis
[(next section)](#data-preprocessing) | [(previous section)](#feature-engineering) | [Table of Contents](#table-of-contents)

---

## 05 - Data Preprocessing
[(next section)](#feature-selection) | [(previous section)](#exploratory-data-analysis) | [Table of Contents](#table-of-contents)

---

## 06 - Feature Selection
[(next section)](#machine-learning-modeling) | [(previous section)](#data-preprocessing) | [Table of Contents](#table-of-contents)

---

## 07 - Machine Learning Modeling
[(next section)](#hyperparameter-tuning) | [(previous section)](#feature-selection) | [Table of Contents](#table-of-contents)

---

## 08 - Hyperparameter Tuning
[(next section)](#understanding-the-error-and-business-performance) | [(previous section)](#machine-learning-modeling) | [Table of Contents](#table-of-contents)

---

## 09 - Understanding the Error and Business Performance
[(next section)](#final-model-deployment) | [(previous section)](#hyperparameter-tuning) | [Table of Contents](#table-of-contents)

---

## 10 - Final Model Deployment
[(next section)](#the-final-solution) | [(previous section)](#understanding-the-error-and-business-performance) | [Table of Contents](#table-of-contents)

---

## 11 - The Final Solution
[(next section)](#conclusion) | [(previous section)](#final-model-deployment) | [Table of Contents](#table-of-contents)

--- 

## Conclusion
[(previous section)](#the-final-solution) | [Table of Contents](#table-of-contents) | [Back to Top](#rossmann-stores-sale-prediction)



