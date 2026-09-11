CREATE TABLE actors (
    actor_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    age DATE NOT NULL,
    number_oscars SMALLINT NOT NULL
);

INSERT INTO actors (first_name, last_name, age, number_oscars)
VALUES
    ('Matt', 'Damon', '1970-10-08', 5),
    ('George', 'Clooney', '1961-05-06', 2),
    ('Jennifer', 'Aniston', '1969-02-11', 0),
    ('Harrison', 'Ford', '1942-07-13', 4);

	
INSERT INTO actors (first_name, last_name, age, number_oscars)
VALUES
    ('Tom', 'Hanks', '1956-07-09', 2),
    ('Leonardo', 'DiCaprio', '1974-11-11', 1),
    ('Meryl', 'Streep', '1949-06-22', 3),
    ('Denzel', 'Washington', '1954-12-28', 2),
    ('Natalie', 'Portman', '1981-06-09', 1),
    ('Brad', 'Pitt', '1963-12-18', 2),
    ('Emma', 'Stone', '1988-11-06', 2),
    ('Morgan', 'Freeman', '1937-06-01', 1),
    ('Kate', 'Winslet', '1975-10-05', 1);

SELECT COUNT(*)	AS number_of_actors
FROM actors;

INSERT INTO actors (first_name, last_name, age, number_oscars)
VALUES
    ();
