[./model_rf.py](./model_rf.py) > [./input/20240110_cubist_input_spi1](./input/20240110_cubist_input_spi1)을 입력자료로 하여 마지막 열 예측

- 줄리안일별 RF 모델은 output/rfs에 ```.json```으로 저장


[./load_model.py](./load_model.py) > 훈련한 RF 모델에 [./input/output_건천리_추가.xlsx](./input/output_건천리_추가.xlsx)의 ```'DEM', 'LC', 'AWC', 'ECO', 'SSG', 'SPI1'```을 입력자료로 하여 안성시(FID 10) ```SDI``` 예측
