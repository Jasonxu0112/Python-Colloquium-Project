class User:
    def __init__(self, user_id, name, group_size, preferred_environment, budget):
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        self.preferred_environment = preferred_environment
        self.budget = budget

    def match_property_listing(self, property_listing):
        if self.budget >= property_listing.price_per_night:
            return True
        return False