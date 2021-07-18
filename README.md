# SOC_stock_prediction
Machine learning are being widely used in financial institution such as high-frequency trading, portfolio optimization, fraud detection etc.
In this project we model a gerenrative adversarial metwrok with Gated Recurrent Units (GRU) and LSTM as generator to generate stock price and CNN as discriminator.
I will be using Goldman sachs stock closing price using features like S&P 500 index, NASDAQ Composite index, us dollar index, etc to predict feature. 

## DATA
**correlated assets** \
For the data we incorporate as much information about the company's performance as well as the market sentiment as possible.
we use correlated asstes like indices commodities price , other big companies etc.
Close, Volume, NASDAQ, NYSE, S&P 500, Crude Oil, Gold, USD index, Morgan stanley, MA7, MA21, MACD, 20SD, upper_band, lower_band, EMA, log momentum, absolute of 3 comps, angle of 3 comps, absolute of 6 comps, angle of 6 comps, absolute of 9 comps, angle of 9 comps etc. 
there are 1617 observations used in the dataset. The train data and test data are split into 0.7:0.3.

![image](https://user-images.githubusercontent.com/70603282/126057225-f8ab676d-8e98-4a99-9a2e-8af932f99ebe.png)

**techincal indicators** \
We add 7 and 21 day moving avergae , momentum bollinger bands macd etc 

![image](https://user-images.githubusercontent.com/70603282/126057276-8e54304c-4549-48b3-af94-42558fed17a4.png)


**fourier transforms** \
In order to get the long trem and short term trends we use fourier transforms to create approximations of real stock movements.

![image](https://user-images.githubusercontent.com/70603282/126057278-1656f528-e7fd-426e-ad7b-5dd4c366bc6e.png)


**ARIMA** \
ARIMA is a pre neural net techniques for predicting time series data.

**Feature importance** \
Create feature importance.we use XGBoost) for feature importance.

![image](https://user-images.githubusercontent.com/70603282/126057296-08a113be-c7ae-4063-a2aa-9294a84ba0b2.png)

## GAN

we will use GANs with LSTM and GRU model for generator and CNN for discriminator
the generator uses random data to generate data as close to the real data and the discriminator classifies whether data is generated using generator or is the real data

**generator** \
for generator we use LSTM and GRU as RNN. The biggest differences between the two are: 1) GRU has 2 gates (update and reset) and LSTM has 4 (update, input, forget, and output), 2) LSTM maintains an internal memory state, while GRU doesn’t, and 3) LSTM applies a nonlinearity (sigmoid) before the output gate, GRU doesn’t.

**discriminator** \
we use CNN as discriminator as CNN are powerful at extracting features like small and and bigger trends as it works well with spatial data. So it should work with time series data.
 
 **W_GAN with GRU training generator with three days and generating 1 day** \
 ![image](https://user-images.githubusercontent.com/70603282/126057963-dae89a75-9e05-4016-87c4-c0de3cc2a207.png)

![image](https://user-images.githubusercontent.com/70603282/126057985-eaa1a065-a1fc-431a-b6e4-28ca75b29266.png)

**W_GAN with GRU training generator with seven days and generating 1 day** \
![image](https://user-images.githubusercontent.com/70603282/126058049-82897bdb-c143-41be-a8a3-0676c2f342ae.png)

![image](https://user-images.githubusercontent.com/70603282/126058059-ff3d86c0-dfd6-4300-ad1b-307c2312bd2c.png)

**W_GAN with LSTM training generator with three days and generating 1 day** \
![image](https://user-images.githubusercontent.com/70603282/126058101-4bcb8a33-0f0f-44a9-8103-14995b5f9401.png)

![image](https://user-images.githubusercontent.com/70603282/126058105-386d9eed-dc2f-4c46-9d43-df393fdbb193.png)

**W_GAN with LSTM training generator with seven days and generating 1 day** \
![image](https://user-images.githubusercontent.com/70603282/126058085-6c34b7a7-71e3-43a0-855c-392031fd08a2.png)

![image](https://user-images.githubusercontent.com/70603282/126058090-49758aab-eabb-403e-bbeb-12bd88f1d940.png)

## Evaluation
RSME result for different models on test data

|             | 3 days - 1 day | 7 days - 1 days |
| ----------- | -------------- | --------------- |
| GAN (LSTM)  | 26.0821        |       12.8483   |
| GAN (GRU)   | 8.8541         |      13.20      |

## Next
we can scrape news headline and use NLP to get the news sentiment as data in future
we can add autoencoders to extract features and use PCA analysis to reduce the number of features in neural network

