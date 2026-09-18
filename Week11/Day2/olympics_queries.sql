-- ============================================================
-- OLYMPICS SQL EXERCISES
-- Exercise 1 and Exercise 2
-- ============================================================

-- ============================================================
-- EXERCISE 1
-- ============================================================

-- Task 1:
-- Find the average age of competitors who have won at least one medal,
-- grouped by medal type. Use a correlated subquery.

SELECT
    m.medal_name,
    (
        SELECT AVG(gc.age)
        FROM games_competitor gc
        WHERE gc.age IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM competitor_event ce
              WHERE ce.competitor_id = gc.id
                AND ce.medal_id = m.id
          )
    ) AS average_age
FROM medal m
WHERE m.medal_name IN ('Gold', 'Silver', 'Bronze');


-- Task 2:
-- Identify the top 5 regions with the highest number of unique competitors
-- who participated in more than 3 different events.
-- Use nested subqueries to filter and aggregate the data.

SELECT
    nr.region_name,
    COUNT(DISTINCT q.person_id) AS unique_competitors
FROM (
    SELECT gc.person_id
    FROM games_competitor gc
    WHERE gc.id IN (
        SELECT ce.competitor_id
        FROM competitor_event ce
        GROUP BY ce.competitor_id
        HAVING COUNT(DISTINCT ce.event_id) > 3
    )
) AS q
JOIN person_region pr
    ON q.person_id = pr.person_id
JOIN noc_region nr
    ON pr.region_id = nr.id
GROUP BY nr.id, nr.region_name
ORDER BY unique_competitors DESC
LIMIT 5;


-- Task 3:
-- Create a temporary table containing the total number of medals won
-- by each competitor and keep only competitors who won more than 2 medals.
-- Use a subquery to aggregate the data.

DROP TABLE IF EXISTS temp_competitor_medals;

CREATE TEMP TABLE temp_competitor_medals AS
SELECT *
FROM (
    SELECT
        gc.person_id,
        p.full_name,
        COUNT(*) AS total_medals
    FROM games_competitor gc
    JOIN competitor_event ce
        ON ce.competitor_id = gc.id
    JOIN medal m
        ON ce.medal_id = m.id
    JOIN person p
        ON gc.person_id = p.id
    WHERE m.medal_name IN ('Gold', 'Silver', 'Bronze')
    GROUP BY gc.person_id, p.full_name
) AS medal_totals
WHERE total_medals > 2;

SELECT *
FROM temp_competitor_medals
ORDER BY total_medals DESC;


-- Task 4:
-- Create a temporary table for analysis and use a subquery within DELETE
-- to remove competitors who have not won any medals.

DROP TABLE IF EXISTS temp_competitors_analysis;

CREATE TEMP TABLE temp_competitors_analysis AS
SELECT DISTINCT
    gc.person_id,
    p.full_name
FROM games_competitor gc
JOIN person p
    ON gc.person_id = p.id;

DELETE FROM temp_competitors_analysis
WHERE person_id NOT IN (
    SELECT DISTINCT gc.person_id
    FROM games_competitor gc
    JOIN competitor_event ce
        ON ce.competitor_id = gc.id
    JOIN medal m
        ON ce.medal_id = m.id
    WHERE m.medal_name IN ('Gold', 'Silver', 'Bronze')
);

SELECT *
FROM temp_competitors_analysis;


-- ============================================================
-- EXERCISE 2
-- ============================================================

-- Task 1:
-- Update missing competitor heights using the average height of competitors
-- from the same region. Use a correlated subquery inside UPDATE.

UPDATE person AS p
SET height = (
    SELECT AVG(p2.height)
    FROM person p2
    JOIN person_region pr2
        ON p2.id = pr2.person_id
    WHERE p2.height IS NOT NULL
      AND pr2.region_id IN (
          SELECT pr1.region_id
          FROM person_region pr1
          WHERE pr1.person_id = p.id
      )
)
WHERE p.height IS NULL;


-- Task 2:
-- Insert competitors who participated in more than one event in the same Games
-- into a temporary table and store their total number of events.
-- Use nested subqueries.

DROP TABLE IF EXISTS temp_multi_event_competitors;

CREATE TEMP TABLE temp_multi_event_competitors (
    competitor_id INTEGER,
    person_id INTEGER,
    games_id INTEGER,
    total_events INTEGER
);

INSERT INTO temp_multi_event_competitors
    (competitor_id, person_id, games_id, total_events)
SELECT
    gc.id,
    gc.person_id,
    gc.games_id,
    event_counts.total_events
FROM games_competitor gc
JOIN (
    SELECT *
    FROM (
        SELECT
            competitor_id,
            COUNT(DISTINCT event_id) AS total_events
        FROM competitor_event
        GROUP BY competitor_id
    ) AS counts
    WHERE total_events > 1
) AS event_counts
    ON gc.id = event_counts.competitor_id;

SELECT *
FROM temp_multi_event_competitors
ORDER BY total_events DESC;


-- Task 3:
-- Identify regions where the average number of medals won per competitor
-- is greater than the overall average.

SELECT
    nr.region_name,
    AVG(region_medals.medal_count) AS avg_medals_per_competitor
FROM (
    SELECT
        gc.person_id,
        SUM(
            CASE
                WHEN m.medal_name IN ('Gold', 'Silver', 'Bronze') THEN 1
                ELSE 0
            END
        ) AS medal_count
    FROM games_competitor gc
    LEFT JOIN competitor_event ce
        ON gc.id = ce.competitor_id
    LEFT JOIN medal m
        ON ce.medal_id = m.id
    GROUP BY gc.person_id
) AS region_medals
JOIN person_region pr
    ON region_medals.person_id = pr.person_id
JOIN noc_region nr
    ON pr.region_id = nr.id
GROUP BY nr.id, nr.region_name
HAVING AVG(region_medals.medal_count) > (
    SELECT AVG(overall_medals.medal_count)
    FROM (
        SELECT
            gc2.person_id,
            SUM(
                CASE
                    WHEN m2.medal_name IN ('Gold', 'Silver', 'Bronze') THEN 1
                    ELSE 0
                END
            ) AS medal_count
        FROM games_competitor gc2
        LEFT JOIN competitor_event ce2
            ON gc2.id = ce2.competitor_id
        LEFT JOIN medal m2
            ON ce2.medal_id = m2.id
        GROUP BY gc2.person_id
    ) AS overall_medals
)
ORDER BY avg_medals_per_competitor DESC;


-- Task 4:
-- Create a temporary table to track competitors' participation by season
-- and identify competitors who participated in both Summer and Winter Games.

DROP TABLE IF EXISTS temp_season_participation;

CREATE TEMP TABLE temp_season_participation AS
SELECT
    gc.person_id,
    p.full_name,
    g.season,
    COUNT(DISTINCT gc.games_id) AS games_count
FROM games_competitor gc
JOIN games g
    ON gc.games_id = g.id
JOIN person p
    ON gc.person_id = p.id
GROUP BY
    gc.person_id,
    p.full_name,
    g.season;

SELECT
    person_id,
    full_name
FROM temp_season_participation
WHERE season IN ('Summer', 'Winter')
GROUP BY person_id, full_name
HAVING COUNT(DISTINCT season) = 2
ORDER BY full_name;
