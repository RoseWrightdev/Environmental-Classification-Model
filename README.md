# Air Quality and Pollution Assessment Classification Model

## Key features of the dataset
- Temperature (°C): Average temperature of the region.

- Humidity (%): Relative humidity recorded in the region.

- PM2.5 Concentration (µg/m³): Fine particulate matter levels.

- PM10 Concentration (µg/m³): Coarse particulate matter levels.

- NO2 Concentration (ppb): Nitrogen dioxide levels.

- SO2 Concentration (ppb): Sulfur dioxide levels.

- CO Concentration (ppm): Carbon monoxide levels.

- Proximity to Industrial Areas (km): Distance to the nearest industrial zone.

- Population Density (people/km²): Number of people per square kilometer in the region.

## Target Variable: Air Quality Levels
- Good: Clean air with low pollution levels.

- Moderate: Acceptable air quality but with some pollutants present.

- Poor: Noticeable pollution that may cause health issues for sensitive groups.

- Hazardous: Highly polluted air poses serious health risks to the population.

## Dataset
https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment

### Correlation Matrix
![image](https://github.com/user-attachments/assets/c2005c81-5673-4537-9091-cc1304a579c9)



# Model

```
# Hyperparameters
batch_size = 256
hidden_units = 1028
dropout = 0.55
output_size = 4
input_size = len(df.columns)-1
epochs = 1000
model = tf.keras.Sequential()

# Structure
model.add(tf.keras.Input(shape=(input_size,))) # input 

model.add(Dense(hidden_units))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units//2))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units//4))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units//8))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout))

model.add(Dense(output_size, activation='softmax')) # output 

model.summary()
```
![image](https://github.com/user-attachments/assets/0d8dccf9-7a65-4a25-8cf3-90784f27af86)

## Results
![image](https://github.com/user-attachments/assets/c92139c5-c281-4f48-9b19-e50eb0bc0899)

![image](https://github.com/user-attachments/assets/bbca14b4-a1da-4b0e-804c-9868439c0911)
