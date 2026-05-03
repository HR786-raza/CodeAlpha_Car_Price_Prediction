import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)


def validate_numeric(value: Any, name: str) -> float:
    return float(value)


def validate_integer(value: Any, name: str) -> int:
    return int(value)


def validate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Present_Price": validate_numeric(data["Present_Price"], "Present_Price"),
        "Driven_kms": validate_numeric(data["Driven_kms"], "Driven_kms"),
        "Owner": validate_integer(data["Owner"], "Owner"),
        "Car_Age": validate_integer(data["Car_Age"], "Car_Age"),
        "Fuel_Type": data["Fuel_Type"],
        "Selling_type": data["Selling_type"],
        "Transmission": data["Transmission"],
    }


def format_price(price: float) -> str:
    return f"{price:.2f} Lakhs"


def log_prediction(data, pred):
    logging.info(f"{data} -> {pred}")