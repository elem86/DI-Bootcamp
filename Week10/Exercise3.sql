--1.
SELECT *
FROM language;

--2.
SELECT film.title, film.description, language.name
FROM film
JOIN language ON film.language_id = language.language_id;

--3.
SELECT film.title, film.description, language.name
FROM language
LEFT JOIN film ON language.language_id = film.language_id;

-- 4.
CREATE TABLE new_film (
	id SERIAL PRIMARY KEY,
	name VARCHAR(50) NOT NULL);

INSERT INTO new_film (name)
VALUES
    ('Love and love'),
    ('Clooney and the man'),
    ('Aniston and the stove'),
    ('Ford and the car');

-- 5.
CREATE TABLE customer_review (
    review_id SERIAL PRIMARY KEY,
    film_id INTEGER NOT NULL,
    language_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    score INTEGER CHECK (score BETWEEN 1 AND 10),
    review_text TEXT,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_review_film
        FOREIGN KEY (film_id)
        REFERENCES new_film(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_review_language
        FOREIGN KEY (language_id)
        REFERENCES language(language_id)
);

--6.
INSERT INTO customer_review
    (film_id, language_id, title, score, review_text)
VALUES
    (1, 1, 'Great movie', 9, 'I really enjoyed this film.'),
    (2, 1, 'Pretty good', 7, 'It was entertaining and well made.');

--7.
DELETE FROM new_film
WHERE id = 1;