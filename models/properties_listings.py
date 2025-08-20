class Property:
    def __init__(self, property_id, location, type_, price_per_night, features=None, tags=None):
        self._property_id = property_id
        self._location = location
        self._type = type_
        self._price_per_night = price_per_night
        self._features = features if features is not None else []
        self._tags = tags if tags is not None else []

    # Getter and Setter for property_id
    @property
    def property_id(self):
        return self._property_id

    @property_id.setter
    def property_id(self, property_id):
        self._property_id = property_id

    # Getter and Setter for location
    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._location = location

    # Getter and Setter for type
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type_):
        self._type = type_

    # Getter and Setter for price_per_night
    @property
    def price_per_night(self):
        return self._price_per_night

    @price_per_night.setter
    def price_per_night(self, price_per_night):
        if price_per_night < 0:
            raise ValueError("Price cannot be negative.")
        self._price_per_night = price_per_night

    # Getter and Setter for features
    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        # self._features = features
        if not isinstance(features, list):
            raise TypeError("Features must be a list.")

    # Getter and Setter for tags
    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags):
        # self._tags = tags
        if not isinstance(tags, list):
            raise TypeError("Tags must be a list.")
