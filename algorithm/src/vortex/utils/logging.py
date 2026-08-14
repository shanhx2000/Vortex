import logging

def setup_logging(filename, level=logging.INFO):
    # add a timestamp to filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{filename}-{timestamp}.log"
    logging.basicConfig(
        filename=log_filename,          # file name
        filemode="a",                # "a" = append, "w" = overwrite
        level=level,          # minimum level to log
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Application started")
    logging.error("Something went wrong!")
