from time import sleep
from modules.util import jsonify, time_gap, save_json
from modules.crawl import reviewAPI

# Constant Setting
HAS_NEXT_PAGE = True
NEXT_CURSOR = ""
crawled_reviews = {}

#  Crawling
while HAS_NEXT_PAGE:
    # API Requests
    next_reviews = (
        reviewAPI().get(uuid="0f101f8c-ec09-39c4-9be0-2f9cc464d332", cursor=NEXT_CURSOR, page_count=20).text
    )
    next_reviews = jsonify(next_reviews)

    # Save as Dictionary
    if len(crawled_reviews) == 0:
        crawled_reviews = next_reviews
    else:
        crawled_reviews["reviews"].extend(next_reviews["reviews"])

    # Next Request Conditions
    HAS_NEXT_PAGE = next_reviews["pageInfo"]["hasNextPage"]
    NEXT_CURSOR = (
        next_reviews["pageInfo"]["endCursor"]
        if ("endCursor" in next_reviews["pageInfo"].keys()) & HAS_NEXT_PAGE
        else ""
    )

    # Time Delay
    sleep(time_gap(mu=4))

# Save Dictionary to JSON files
save_json("./database/venom_review.json", crawled_reviews)
