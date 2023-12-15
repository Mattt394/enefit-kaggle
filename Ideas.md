# Feature Engineering Ideas

Predicted energy price is for yesterday's price, as they can't get today's. Might be useful to give, along with the hour's price, also a moving average and a momentum indicator to indicate price movement direction. But, it also might be that electricity price is not that predictive.

Also might want to do something for gas prices.

#### Historical Weather
No weather stations map to county id 12. Create an average weather across all of estonia mapping to fill the na values for county id 12.
It might be interesting to include average weather for every row anyways.
I'm giving each row the weather from 24h previous. BUT, it might be better to give each row the last weather available from the data_block_id. i.e., the historical weather for hour_part 23 or whatever it is.