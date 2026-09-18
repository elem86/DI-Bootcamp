--1. 

-- DROP TABLE IF EXISTS temp_both_seasons_medalists;

-- CREATE TEMP TABLE temp_both_seasons_medalists AS
-- SELECT
--     gc.person_id,
--     p.full_name,
--     SUM(CASE WHEN g.season = 'Summer' THEN 1 ELSE 0 END) AS summer_medals,
--     SUM(CASE WHEN g.season = 'Winter' THEN 1 ELSE 0 END) AS winter_medals
-- FROM games_competitor gc
-- JOIN games g
--     ON gc.games_id = g.id
-- JOIN competitor_event ce
--     ON ce.competitor_id = gc.id
-- JOIN medal m
--     ON ce.medal_id = m.id
-- JOIN person p
--     ON gc.person_id = p.id
-- WHERE m.medal_name IN ('Gold', 'Silver', 'Bronze')
-- GROUP BY gc.person_id, p.full_name
-- HAVING SUM(CASE WHEN g.season = 'Summer' THEN 1 ELSE 0 END) > 0
--    AND SUM(CASE WHEN g.season = 'Winter' THEN 1 ELSE 0 END) > 0;

-- SELECT *
-- FROM temp_both_seasons_medalists
-- ORDER BY summer_medals + winter_medals DESC;


--2. 

-- CREATE TEMP TABLE temp_two_sport_medalists AS
-- SELECT
--     person_id,
--     full_name,
--     total_medals
-- FROM (
--     SELECT
--         gc.person_id,
--         p.full_name,
--         COUNT(*) AS total_medals
--     FROM games_competitor gc
--     JOIN competitor_event ce
--         ON gc.id = ce.competitor_id
--     JOIN medal m
--         ON ce.medal_id = m.id
--     JOIN event e
--         ON ce.event_id = e.id
--     JOIN person p
--         ON gc.person_id = p.id
--     WHERE m.medal_name IN ('Gold', 'Silver', 'Bronze')
--     GROUP BY gc.person_id, p.full_name
--     HAVING COUNT(DISTINCT e.sport_id) = 2
-- ) AS medal_counts
-- ORDER BY total_medals DESC
-- LIMIT 3;

-- SELECT *
-- FROM temp_two_sport_medalists;


--3.
-- SELECT
--     nr.region_name,
--     SUM(best_event.max_medals) AS total_medals
-- FROM (
--     SELECT
--         person_id,
--         MAX(medal_count) AS max_medals
--     FROM (
--         SELECT
--             gc.person_id,
--             ce.event_id,
--             COUNT(*) AS medal_count
--         FROM games_competitor gc
--         JOIN competitor_event ce
--             ON gc.id = ce.competitor_id
--         JOIN medal m
--             ON ce.medal_id = m.id
--         WHERE m.medal_name IN ('Gold', 'Silver', 'Bronze')
--         GROUP BY gc.person_id, ce.event_id
--     ) AS event_medals
--     GROUP BY person_id
-- ) AS best_event
-- JOIN person_region pr
--     ON best_event.person_id = pr.person_id
-- JOIN noc_region nr
--     ON pr.region_id = nr.id
-- GROUP BY nr.id, nr.region_name
-- ORDER BY total_medals DESC
-- LIMIT 5;


--4.
-- DROP TABLE IF EXISTS temp_no_medal_competitors;

-- CREATE TEMP TABLE temp_no_medal_competitors AS
-- SELECT
--     gc.person_id,
--     p.full_name,
--     COUNT(DISTINCT gc.games_id) AS games_participated
-- FROM games_competitor gc
-- JOIN person p
--     ON gc.person_id = p.id
-- LEFT JOIN competitor_event ce
--     ON gc.id = ce.competitor_id
-- LEFT JOIN medal m
--     ON ce.medal_id = m.id
-- GROUP BY gc.person_id, p.full_name
-- HAVING COUNT(DISTINCT gc.games_id) > 3
--    AND SUM(
--        CASE
--            WHEN m.medal_name IN ('Gold', 'Silver', 'Bronze') THEN 1
--            ELSE 0
--        END
--    ) = 0;
SELECT *
FROM temp_no_medal_competitors
ORDER BY games_participated DESC;
