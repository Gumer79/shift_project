import os
import argparse
import pandas as pd
import numpy as np
import logging
import requests
from sqlalchemy import create_engine, text

from dotenv import load_dotenv

load_dotenv()


def get_weather_data(start_date: str, end_date: str):

    """Функция для получения данных из API."""

    # URL-адрес API
    url = "https://api.open-meteo.com/v1/forecast"

    # Параметры запроса: что именно мы хотим получить от API
    params = {
        "latitude": 55.0344,
        "longitude": 82.9434,
        "daily": "sunrise,sunset,daylight_duration",
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
            "temperature_80m,temperature_120m,wind_speed_10m,wind_speed_80m,"
            "wind_direction_10m,wind_direction_80m,visibility,evapotranspiration,"
            "weather_code,soil_temperature_0cm,soil_temperature_6cm,rain,showers,snowfall"
        ),
        "timezone": "auto",
        "timeformat": "unixtime",
        "wind_speed_unit": "kn",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "start_date": start_date,
        "end_date": end_date,
    }

    logging.info("Пробую выгрузить данные из api.open-meteo.com ...")
    response = requests.get(url, params=params, timeout=10)

    # Попытка загрузить данные с 5 повторами в случае неудачи
    repeat = 5
    while repeat > 0:
        try:
            if response.status_code == 200:
                weather_data = response.json()
                logging.info(
                    f"✅ С сайта данные успешно получены c {start_date} по {end_date}."
                )
                print(
                    f"✅ С сайта данные успешно получены c {start_date} по {end_date}."
                )
                return weather_data
            else:
                repeat -= 1

        except exception as e:
            logging.error(e, exc_info=True)
            logging.info(f"❌ Ошибка после нескольких попыток: {e}")


def transform(data: dict):
    """Функция для трансформации данных в pandas df."""
    df = pd.DataFrame(data)

    # Создаем DataFrame для ежедневных данных
    daily_df = pd.DataFrame(
        df["daily"][["sunrise", "sunset", "daylight_duration"]].to_dict()
    )

    daily_df["sunrise"] = (
        pd.to_datetime(daily_df["sunrise"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Etc/GMT-7")
    )

    daily_df["sunset"] = (
        pd.to_datetime(daily_df["sunset"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Etc/GMT-7")
    )

    # Создаем столбцы с временем в стандартном формате ISO для итоговой таблицы.
    daily_df["sunrise_iso"] = (
        daily_df["sunrise"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    daily_df["sunset_iso"] = (
        daily_df["sunset"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # Создаем DataFrame для почасовых данных
    data_dict = df["hourly"].to_dict()

    df_hourly = pd.DataFrame(data_dict)
    df_hourly["time"] = pd.to_datetime(df_hourly["time"], unit="s")
    df_hourly["time"] = (
        df_hourly["time"].dt.tz_localize("UTC").dt.tz_convert("Etc/GMT-7")
    )

    df_hourly["date"] = df_hourly["time"].dt.date
    df_hourly = df_hourly[
        [
            "time",
            "date",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "temperature_80m",
            "temperature_120m",
            "wind_speed_10m",
            "wind_speed_80m",
            "wind_direction_10m",
            "wind_direction_80m",
            "visibility",
            "evapotranspiration",
            "weather_code",
            "soil_temperature_0cm",
            "soil_temperature_6cm",
            "rain",
            "showers",
            "snowfall",
        ]
    ]

    # Объединяем почасовые и ежедневные данные. Теперь у каждой часовой записи есть информация
    # о времени восхода и заката для этого дня.
    df_hourly = df_hourly.merge(
        daily_df,
        how="left",
        left_on=df_hourly["date"],
        right_on=daily_df["sunrise"].dt.date,
        copy=False,
    )

    df_hourly.drop("key_0", axis=1, inplace=True)

    # --- Конвертация единиц измерения ---

    # Список столбцов с температурой в Фаренгейтах
    farengate_cols2_celsius = [
        "temperature_2m",
        "dew_point_2m",
        "apparent_temperature",
        "temperature_80m",
        "temperature_120m",
        "soil_temperature_0cm",
        "soil_temperature_6cm",
    ]

    # Конвертируем в градусы Цельсия по формуле
    for col in farengate_cols2_celsius:
        df_hourly[col] = (df_hourly[col] - 32) * 5 / 9

    # Список столбцов с видимостью в футах
    feet_cols2_m = ["visibility"]

    # Конвертируем в футы по формуле в метры
    for col in feet_cols2_m:
        df_hourly[col] = df_hourly[col] * 0.3048

    # Список столбцов с осадками в дюймах
    inch_cols2_mm = ["evapotranspiration", "rain", "showers", "snowfall"]
    for col in inch_cols2_mm:
        df_hourly[col] = df_hourly[col] * 25.4

    # Список столбцов со скоростью ветра в узлах
    kn_cols2_m_s = ["wind_speed_10m", "wind_speed_80m"]

    # Конвертируем их в метры в секунду
    for col in kn_cols2_m_s:
        df_hourly[col] = df_hourly[col] * 0.514444

        # --- Создание новых признаков ---

    # Создаем маску, чтобы определить, является ли время дневным
    daytime_mask = (df_hourly["time"] >= df_hourly["sunrise"]) & (
        df_hourly["time"] <= df_hourly["sunset"]
    )
    # Создаем столбец 'day_time' со значениями 'day' или 'night'
    df_hourly["day_time"] = np.where(daytime_mask, "day", "night")

    # Рассчитываем продолжительность светового дня в часах
    df_hourly["daylight_hours"] = (
        df_hourly["sunset"] - df_hourly["sunrise"]
    ).dt.total_seconds() / 3600

    return df_hourly


def creat_final_dataframe(df):

    """Функция получает трансформированный df и выводить итоговую таблицу."""

    # 1. Агрегация за 24 часа. Считаем средние (mean) и суммы (sum) для каждой даты.
    all_hours_aggs = df.groupby("date").agg(
        avg_temperature_2m_24h=("temperature_2m", "mean"),
        avg_relative_humidity_2m_24h=("relative_humidity_2m", "mean"),
        avg_dew_point_2m_24h=("dew_point_2m", "mean"),
        avg_apparent_temperature_24h=("apparent_temperature", "mean"),
        avg_temperature_80m_24h=("temperature_80m", "mean"),
        avg_temperature_120m_24h=("temperature_120m", "mean"),
        avg_wind_speed_10m_24h=("wind_speed_10m", "mean"),
        avg_wind_speed_80m_24h=("wind_speed_80m", "mean"),
        avg_visibility_24h=("visibility", "mean"),
        total_rain_24h=("rain", "sum"),
        total_showers_24h=("showers", "sum"),
        total_snowfall_24h=("snowfall", "sum"),
    )

    # 2. Агрегация только за дневное время.
    # Сначала фильтруем DataFrame, оставляя только 'day'

    df_day = df[df["day_time"] == "day"].copy()

    # Считаем средние и суммы только для дневных часов
    daytiime_aggs = df_day.groupby("date").agg(
        avg_temperature_2m_daylight=("temperature_2m", "mean"),
        avg_relative_humidity_2m_daylight=("relative_humidity_2m", "mean"),
        avg_dew_point_2m_daylight=("dew_point_2m", "mean"),
        avg_apparent_temperature_daylight=("apparent_temperature", "mean"),
        avg_temperature_80m_daylight=("temperature_80m", "mean"),
        avg_temperature_120m_daylight=("temperature_120m", "mean"),
        avg_wind_speed_10m_daylight=("wind_speed_10m", "mean"),
        avg_wind_speed_80m_daylight=("wind_speed_80m", "mean"),
        avg_visibility_daylight=("visibility", "mean"),
        total_rain_daylight=("rain", "sum"),
        total_showers_daylight=("showers", "sum"),
        total_snowfall_daylight=("snowfall", "sum"),
    )
    # 3. Агрегация в списки. Собираем все почасовые значения в один список для каждой даты.
    # Это полезно, если нужно сохранить все исходные данные в сжатом виде.

    list_aggs = df.groupby("date").agg(
        wind_speed_10m_m_per_s=("wind_speed_10m", list),
        wind_speed_80m_m_per_s=("wind_speed_80m", list),
        temperature_2m_celsius=("temperature_2m", list),
        apparent_temperature_celsius=("apparent_temperature", list),
        temperature_80m_celsius=("temperature_80m", list),
        temperature_120m_celsius=("temperature_120m", list),
        soil_temperature_0cm_celsius=("soil_temperature_0cm", list),
        soil_temperature_6cm_celsius=("soil_temperature_6cm", list),
        rain_mm=("rain", list),
        showers_mm=("showers", list),
        snowfall_mm=("snowfall", list),
    )
    # 4. Агрегация  значений для каждой даты (продолжительность светового дня в часах, восход, закат и т.д.)
    dates_aggs = df.groupby("date").agg(
        daylight_hours=("daylight_hours", "min"),
        sunset_iso=("sunset_iso", "min"),
        sunrise_iso=("sunrise_iso", "min"),
    )

    # Собираем все агрегированные DataFrame в один итоговый
    final_df = all_hours_aggs.join(daytiime_aggs).join(list_aggs).join(dates_aggs)
    return final_df


def get_db_conn():
    """Функция для получения подключения к БД."""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = "localhost"  # or your server's IP address
    db_port = "5432"  # default postgres port
    db_name = os.getenv("DB_NAME")

    connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    engine = create_engine(connection_string)
    print("✅ SQLAlchemy engine created successfully!")
    return engine


def load_to_db(df):

    """Функция выгружает итоговую таблицу в БД.
    Для исключение дубликатов сначала выгружается список дат из БД
    и по ним фильтруется итоговая таблица для выгрузки.
    """
    engine = get_db_conn()
    table = "data"

    exist_dates_query = f"""select distinct date
        from public.{table} order by date desc; """

    exist_dates_df = pd.read_sql(text(exist_dates_query), engine)

    exist_dates = exist_dates_df["date"].to_list()
    df = df[~df.index.isin(exist_dates)]

    if not df.empty:
        df.to_sql(
            table,
            engine,
            if_exists="append",
        )

        print(f"✅ Успешно выгружено {len(df)} строк. :) ")
    else:
        print("После фильтрации нечего сохрянять :/ ")
        print("Данные в БД не сохранены. ")


def save_to_csv(df):
    min_date = df.index.min()
    max_date = df.index.max()

    if min_date == max_date:
        file_name = f"{min_date}.csv"
    else:
        file_name = f"{min_date}_{max_date}.csv"

    df.to_csv("csv_data/" + file_name)
    print(f"✅ Успешно cохранен {file_name} файл {len(df)} строк. :)")


def main(start_date: str, end_date: str, load_type="both"):
    """Основная функция для выгрузки."""

    if load_type in ["db", "csv", "both", None]:
        weather_data = get_weather_data(start_date, end_date)
        if weather_data:
            transformed_df = transform(weather_data)
            final_df = creat_final_dataframe(transformed_df)
            if load_type == "db":
                logging.info("Выгружаю только в БД. ")
                load_to_db(final_df)
            elif load_type == "csv":
                logging.info("Выгружаю только в CSV. ")
                save_to_csv(final_df)
            else:
                logging.info("Выгружаю в CSV и БД. ")

                load_to_db(final_df)
                save_to_csv(final_df)

        else:
            logging.error("API вернул пустой отчет. :(")
            logging.error("Проверьте запрос. ")
    else:
        logging.error(f"Не понял тип выгрузки: {load_type}")
        logging.error(
            "Приложение принимает только такие параметры выгрузки 'db', 'csv', 'both' "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Начальная дата выгрузки в формате YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="Конечная дата выгрузки в формате"
    )
    parser.add_argument(
        "--load_type",
        type=str,
        help="""Принимает тип загрузки файла: "db", "csv", "both. По умолчанияю стоит - 'both'. """,
    )

    args = parser.parse_args()

    main(start_date=args.start_date, end_date=args.end_date, load_type=args.load_type)
