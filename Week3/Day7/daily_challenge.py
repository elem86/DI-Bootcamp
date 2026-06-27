"""
Air management system

Classes implemented: Airline, Airplane, Flight, Airport

Notes:
- Each plane can fly at most once per day.
- Flight dates are datetime.date objects (no time-of-day).

This module contains a small demo at the bottom that exercises the API.
"""

from datetime import date as Date
from bisect import insort
from typing import List, Optional


class Airline:
    def __init__(self, id: str, name: str, planes: Optional[List["Airplane"]] = None):
        self.id = id
        self.name = name
        self.planes: List[Airplane] = planes or []

    def __repr__(self):
        return f"Airline({self.id}, {self.name})"


class Airplane:
    def __init__(self, id: int, current_location: "Airport", company: Airline):
        self.id = id
        self.current_location = current_location
        self.company = company
        self.next_flights: List[Flight] = []

        # register plane with company and airport
        if self not in company.planes:
            company.planes.append(self)
        if self not in current_location.planes:
            current_location.planes.append(self)

    def __repr__(self):
        return (
            f"Airplane({self.id}, {self.company.id}, at={self.current_location.city})"
        )

    def fly(self, destination: "Airport") -> Optional["Flight"]:
        """Find the earliest scheduled flight for this airplane to `destination`, take off and land it.

        Returns the Flight if executed, else None.
        """
        for flight in list(self.next_flights):
            if flight.destination is destination:
                flight.take_off()
                flight.land()
                return flight
        return None

    def location_on_date(self, target_date: Date) -> "Airport":
        """Return where the plane will be on target_date.

        Simulate going through `next_flights` in order and apply flights whose date <= target_date.
        """
        loc = self.current_location
        for fl in sorted(self.next_flights, key=lambda f: f.date):
            if fl.date <= target_date:
                loc = fl.destination
            else:
                break
        return loc

    def available_on_date(self, target_date: Date, location: "Airport") -> bool:
        """A plane is available on `target_date` from `location` if:
        - It will be in `location` on that date, and
        - It has no other flight scheduled on that same date (one flight per day rule).
        """
        if self.location_on_date(target_date) is not location:
            return False
        for fl in self.next_flights:
            if fl.date == target_date:
                return False
        return True


class Flight:
    def __init__(
        self, date: Date, destination: "Airport", origin: "Airport", plane: Airplane
    ):
        self.date = date
        self.destination = destination
        self.origin = origin
        self.plane = plane
        # id encoded as destinationCity + airlineCode + YYYYMMDD
        self.id = f"{destination.city}{plane.company.id}{date.strftime('%Y%m%d')}"

    def __repr__(self):
        return f"Flight({self.id}, {self.origin.city}->{self.destination.city} on {self.date}, plane={self.plane.id})"

    def take_off(self):
        """Remove plane from origin airport's planes list and keep state consistent."""
        if self.plane in self.origin.planes:
            self.origin.planes.remove(self.plane)

    def land(self):
        """Set plane location to destination and add it to destination planes list."""
        self.plane.current_location = self.destination
        if self.plane not in self.destination.planes:
            self.destination.planes.append(self.plane)


class Airport:
    def __init__(self, city: str):
        self.city = str(city)
        self.planes: List[Airplane] = []
        self.scheduled_departures: List[Flight] = []
        self.scheduled_arrivals: List[Flight] = []

    def __repr__(self):
        return f"Airport({self.city})"

    def _insert_sorted(self, lst: List[Flight], flight: Flight):
        # keep flights sorted by date
        insort(lst, (flight.date, flight))

    def _extract_sorted(self, lst: List[tuple]) -> List[Flight]:
        return [f for (_d, f) in lst]

    def schedule_flight(
        self,
        destination: "Airport",
        flight_date: Date,
        plane: Optional[Airplane] = None,
    ) -> Flight:
        """Schedule a flight from this airport to `destination` on `flight_date`.

        If `plane` is not provided, finds any plane currently at this airport that is available on that date.
        Raises ValueError if no plane is available.
        """
        # choose plane
        chosen = plane
        if chosen is None:
            for p in list(self.planes):
                if p.available_on_date(flight_date, self):
                    chosen = p
                    break

        if chosen is None:
            raise ValueError("No available plane at this airport for that date")

        # create flight and register it
        flight = Flight(flight_date, destination, self, chosen)

        # add to plane's schedule (keep sorted by date)
        chosen.next_flights.append(flight)
        chosen.next_flights.sort(key=lambda f: f.date)

        # add to scheduled departures / arrivals lists (sorted)
        self.scheduled_departures.append(flight)
        self.scheduled_departures.sort(key=lambda f: f.date)

        destination.scheduled_arrivals.append(flight)
        destination.scheduled_arrivals.sort(key=lambda f: f.date)

        return flight

    def info(self, start_date: Date, end_date: Date) -> List[str]:
        """Return a list of strings describing scheduled departures from start_date to end_date inclusive."""
        out = []
        for fl in self.scheduled_departures:
            if start_date <= fl.date <= end_date:
                out.append(
                    f"{fl.date}: {fl.id} {fl.origin.city}->{fl.destination.city} plane={fl.plane.id}"
                )
        return out


# Demo / simple tests
if __name__ == "__main__":
    from datetime import date, timedelta

    # create airports
    nyc = Airport("NYC")
    lax = Airport("LAX")

    # create airline and planes
    aa = Airline("AA", "American Air")
    p1 = Airplane(101, nyc, aa)
    p2 = Airplane(102, lax, aa)

    today = date.today()
    tomorrow = today + timedelta(days=1)

    # schedule a flight from NYC->LAX today using an available plane at NYC
    f1 = nyc.schedule_flight(lax, today)
    print("Scheduled:", f1)

    # schedule flight from LAX->NYC tomorrow
    f2 = lax.schedule_flight(nyc, tomorrow)
    print("Scheduled:", f2)

    # show info for NYC for next two days
    print("NYC departures:")
    for line in nyc.info(today, tomorrow):
        print(" ", line)

    # where will p1 be tomorrow?
    print(p1, "location on", tomorrow, "->", p1.location_on_date(tomorrow).city)

    # execute p1 flight to LAX
    executed = p1.fly(lax)
    print("Executed flight:", executed)
    print("p1 current location after flight:", p1.current_location.city)
