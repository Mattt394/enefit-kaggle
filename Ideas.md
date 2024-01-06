# Feature Engineering Ideas

Predicted energy price is for yesterday's price, as they can't get today's. Might be useful to give, along with the hour's price, also a moving average and a momentum indicator to indicate price movement direction. But, it also might be that electricity price is not that predictive.

Also might want to do something for gas prices.

Often, they give us the prices for the forecast the day BEFORE. Why don't I create a little LGBM to predict the prices and forecast for the next day? Then I can use those predictions as features in my model :)

Features:
- lags of various lengths
- sum of all product type targets for the same time, county, business type, and consumption type.
- sum of all product types across estonia (i.e. don't group by county)
- create average weather forecasts for all of estonia
- create average weather forecasts for each county
- create average historical weather for all of estonia
- create average historical weather for each county
- For weather features, create 7 day lag
- Target add 2, 3, 4, 5, 6, 7, 14 day lags, an every day in between. Plus maybe last 5 hours too?
- Plus lags for sum of product types
- Add logs for cols with outliers. One kaggle notebook uses:
    # https://www.kaggle.com/code/ruiyaoyang/lb-66-92#Data-transformation
    ['installed_capacity', 'euros_per_mwh', 'temperature_fcast_mean', 'dewpoint_fcast_mean',
        'cloudcover_high_fcast_mean', 'cloudcover_low_fcast_mean', 'cloudcover_mid_fcast_mean', 'cloudcover_total_fcast_mean',
        '10_metre_u_wind_component_fcast_mean', '10_metre_v_wind_component_fcast_mean', 'direct_solar_radiation_fcast_mean',
        'snowfall_fcast_mean', 'total_precipitation_fcast_mean', 'temperature_fcast_mean_by_county', 'dewpoint_fcast_mean_by_county',
        'cloudcover_high_fcast_mean_by_county', 'cloudcover_low_fcast_mean_by_county', 'cloudcover_mid_fcast_mean_by_county',
        'cloudcover_total_fcast_mean_by_county', '10_metre_u_wind_component_fcast_mean_by_county', '10_metre_v_wind_component_fcast_mean_by_county',
        'surface_solar_radiation_downwards_fcast_mean_by_county', 'snowfall_fcast_mean_by_county', 'total_precipitation_fcast_mean_by_county',
        'rain_hist_mean', 'snowfall_hist_mean', 'windspeed_10m_hist_mean_by_county', 'target_2_days_ago', 'target_3_days_ago',
        'target_4_days_ago', 'target_5_days_ago', 'target_6_days_ago', 'target_7_days_ago', 'target_mean', 'target_std']

- I will need to check out which cols I think have outliers as well



#### Historical Weather
No weather stations map to county id 12. Create an average weather across all of estonia mapping to fill the na values for county id 12.
It might be interesting to include average weather for every row anyways.
I'm giving each row the weather from 24h previous. BUT, it might be better to give each row the last weather available from the data_block_id. i.e., the historical weather for hour_part 23 or whatever it is.

### Time Series
Have a look at each time series and check out the scikit-learn libraries

Have a look at training after each day



Ideas to try before submitting:
- ~Try removing features that don't add much~ - Didn't work :(
- ~Do separate models for production and consumption~ - separate models didn't work, but a separate model for producer and a joint model for consumption works - Actually, for my hyperparams at least, doesn't seem to :(
- ~Try linear_trees~ - Definitely didn't work, approached infinite loss....
- Try Ensemble - VotingRegressor/AdaboostRegressor/StackingRegressor/BaggingRegressor
    - Try ensembling with mlens
- ~Try target/installed_capacity for producer and target/eic_count for consumer~
- Try implementing retraining every 14 days or so (will need a small enough model for that)
    - Current idea is to train one super powerful model - my best ensemble. Then also train a smaller model EVERY SINGLE DAY and average or ensemble the predictions. One model will be more powerful, but be getting slightly out of date, and the other model will be always up to date but be super small.
    - WORKED GREAT!
    - RETRAINING every 14 days gave me 64.9 on last fold.
    - Would retraining every 8 days give better results?
    - Current Smaller model not better, need to try again. Or maybe it would be better over longer time periods? Should try submitting.
    - Will try my big model but retraining every week or two
- New LGBM model - predict 24hrs in advance, instead of just 1. All data from last 24hrs is used as features, plus lag features for targets.


My Final Competition Model Idea:
The idea is that predictions degrade as they get further away from the training period, and in my testing a single model continuously retrained is much better than even an ensemble once you get to 6-months out. So, the backbone of my solution will be a single retrainable model. But that leaves a little bit of performance on the table in the periods closer to the training period. So, I'll use an ensemble that gets continuously heavier weighted towards the single model as time goes on past the training period.
- 1st part - My retrainable single lgbm model with magic parameters
- 2nd part - An ensemble of lgbm models (with magic params?)
           - An ensemble of xgboost models
           - These models are not retrained
- Blending - Blend all these models together with a meta model
           - Give the meta model an iteration variable so it knows how far along the prediction time-period is and it will weight the unretrainable models lower and lower as they get further away from the training period.

Todo:
- Create XGBoost model
    - XGB is so bad!! I'm removing my one hot encoding to see if it works better without it.
- Test if my other datasets are actually any good
    - ~Add deep dive into time series exp weighted features~
    - I have data for current magic params model on cv
    - Need to test magic params on cv for lgbm2 (slightly more lag features + weather features) and lgbm3 (lgbm2 + ewm features) datasets
    
- Add second order target change features skew, etc?
- Test new datasets on my magic hyperparams





Ideas:
- Change importance to gain not splits and redo feature selection
- Do hyperparameter tuning on lgbm
- Test out retraining after each test set prediction
- Do some exploratory analysis
- Try out neural prophet
- Try out transformer time series models









Data Processing - Things to handle
- Missing values
- Joining data
- Handling daylight savings missing targets
- Handling prediction_unit_id's missing rows where no. of customers is <5 so they dropped the rows
- Removing outliers
- Do I need to do any log transforms?




Model Ensemble
- Get statsforecast ensemble of models working. Ideas include:
    - ARIMA (AutoRegressive Integrated Moving Average): This model is well-suited for time series data with trends and seasonality. It can incorporate exogenous variables (X) to improve the forecasting.
    - SARIMAX (Seasonal AutoRegressive Integrated Moving-Average with eXogenous regressors): An extension of the ARIMA model, SARIMAX specifically handles multiple seasonal patterns and includes external regressors.
    - ETS (Error, Trend, Seasonality): This model can be used with exogenous regressors and is effective in capturing complex seasonal patterns.
    - TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, and Seasonal components): TBATS is specifically designed for time series with multiple seasonal periods. It can handle exogenous regressors as well.
    - Dynamic Harmonic Regression: This approach combines harmonic regression for seasonality with ARIMA for the error term. It's suitable for time series with complex seasonal patterns and allows for the inclusion of exogenous variables.
  These can be trained easily and I can retrain them every 24 hours for prediction on the next time step
- Get AutoGluon working to do some auto ml for me