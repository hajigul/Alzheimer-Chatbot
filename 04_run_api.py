import uvicorn

from src.config import API_HOST, API_PORT


def main():
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()