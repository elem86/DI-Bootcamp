import os
from datetime import datetime

try:
    from pyowm.owm import OWM
except ImportError as exc:
    raise ImportError(
        "PyOWM is required for this weather app. Install it with `pip install pyowm`"
    ) from exc


def get_api_key() -> str:
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if api_key:
        return api_key.strip()
    return input("Enter your OpenWeatherMap API key: ").strip()


def create_weather_manager():
    api_key = get_api_key()
    owm = OWM(api_key)
    return owm, owm.weather_manager()


def format_timestamp(ts, tz_name="Asia/Jerusalem") -> str:
    try:
        import pytz
    except ImportError:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    tz = pytz.timezone(tz_name)
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def print_tel_aviv_weather(manager):
    observation = manager.weather_at_place("Tel Aviv,IL")
    weather = observation.weather
    print("--- Tel Aviv Current Weather ---")
    print(f"Status: {weather.detailed_status}")
    temp = weather.temperature("celsius").get("temp")
    feels_like = weather.temperature("celsius").get("feels_like")
    humidity = weather.humidity
    print(f"Temperature: {temp:.1f}°C (feels like {feels_like:.1f}°C)")
    print(f"Humidity: {humidity}%")
    wind = weather.wind().get("speed")
    deg = weather.wind().get("deg")
    print(f"Wind: {wind} m/s, bearing {deg}°")
    sunrise = weather.sunrise_time()
    sunset = weather.sunset_time()
    print(f"Sunrise: {format_timestamp(sunrise)}")
    print(f"Sunset: {format_timestamp(sunset)}")
    return observation


def print_weather_for_place_id(manager, place_name):
    observation = manager.weather_at_place(place_name)
    location = observation.location
    city_id = getattr(location, "id", None)
    print(f"\nFound location: {location.name}, {location.country} (id={city_id})")

    if city_id is not None:
        observation_by_id = manager.weather_at_id(city_id)
        weather = observation_by_id.weather
        print(f"--- Weather for {location.name} by city ID ---")
        print(f"Status: {weather.detailed_status}")
        print(f"Temperature: {weather.temperature('celsius').get('temp'):.1f}°C")
        print(f"Humidity: {weather.humidity}%")
        print(f"Wind: {weather.wind().get('speed')} m/s")
    else:
        print("Could not determine a city id for this location.")
    return observation


def print_forecast_for_place(manager, place_name: str):
    print(f"\n--- 5-day forecast for {place_name} ---")
    try:
        forecast = manager.forecast_at_place(place_name, "3h")
        print_forecast_summary(forecast, f"Forecast at place: {place_name}")
    except Exception as exc:
        print(f"Could not retrieve forecast for {place_name}: {exc}")


def print_forecast_summary(forecast, title: str, count: int = 5):
    print(f"\n{title}")
    daily_forecasts = {}

    def parse_iso_datetime(iso_text: str):
        if "T" in iso_text:
            date_part, time_part = iso_text.split("T")
        else:
            date_part, time_part = iso_text.split(" ")
        time_part = time_part[:8]
        hour = int(time_part.split(":")[0])
        return date_part, hour

    for weather in forecast.forecast:
        iso = weather.reference_time("iso")
        date, hour = parse_iso_datetime(iso)
        current = daily_forecasts.get(date)
        if current is None:
            daily_forecasts[date] = (weather, hour)
            continue

        _, current_hour = current
        if abs(hour - 12) < abs(current_hour - 12):
            daily_forecasts[date] = (weather, hour)

    printed = 0
    for date in sorted(daily_forecasts.keys()):
        weather, _ = daily_forecasts[date]
        dt = weather.reference_time("iso")
        temp = weather.temperature("celsius").get("temp")
        humidity = getattr(weather, "humidity", None)
        status = weather.detailed_status
        humidity_text = f", humidity {humidity}%" if humidity is not None else ""
        print(f"{dt}: {status}, {temp:.1f}°C{humidity_text}")
        printed += 1
        if printed >= count:
            break


def format_aqi_label(aqi: int) -> str:
    labels = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }
    return labels.get(aqi, "Unknown")


def print_air_pollution(owm, location_name: str):
    manager = owm.weather_manager()
    observation = manager.weather_at_place(location_name)
    location = observation.location
    lat = location.lat
    lon = location.lon
    air_manager = owm.airpollution_manager()
    pollution = air_manager.air_quality_at_coords(lat=lat, lon=lon)
    air_quality = getattr(pollution, "air_quality", None)
    if air_quality is None:
        air_quality = pollution
    print(f"\n--- Air pollution for {location_name} ---")
    if hasattr(air_quality, "to_dict"):
        air_dict = air_quality.to_dict()
        aqi = air_dict.pop("aqi", None)
        if aqi is not None:
            print(f"Air Quality Index: {aqi} ({format_aqi_label(aqi)})")

        components = (
            air_dict.get("components")
            if isinstance(air_dict.get("components"), dict)
            else air_dict
        )
        if isinstance(components, dict):
            print("Pollutant concentrations:")
            for key, value in components.items():
                print(f"  {key}: {value}")
        else:
            for key, value in air_dict.items():
                print(f"{key}: {value}")
    else:
        print(air_quality)


def main():
    owm, manager = create_weather_manager()
    tel_aviv_observation = print_tel_aviv_weather(manager)
    print_forecast_for_place(manager, "Tel Aviv,IL")
    print_weather_for_place_id(manager, "Tel Aviv,IL")

    place_input = input("\nEnter a city to get weather and city id: ").strip()
    if place_input:
        print_weather_for_place_id(manager, place_input)
        print_forecast_for_place(manager, place_input)

    print_air_pollution(owm, "Tel Aviv,IL")


if __name__ == "__main__":
    main()
