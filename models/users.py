class User:
    def __init__(self, user_id, name, group_size, preferred_environment, budget):
        self._user_id = user_id
        self._name = name
        self._group_size = group_size
        self._preferred_environment = preferred_environment
        self._budget = budget

    # Getter & Setter for user_id
    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, user_id):
        self._user_id = user_id

    # Getter & Setter for name
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    # Getter & Setter for group_size
    @property
    def group_size(self):
        return self._group_size

    @group_size.setter
    def group_size(self, group_size):
        if group_size <= 0:
            raise ValueError("Group size must be positive.")
        self._group_size = group_size

    # Getter & Setter for preferred_environment
    @property
    def preferred_environment(self):
        return self._preferred_environment

    @preferred_environment.setter
    def preferred_environment(self, preferred_environment):
        if not isinstance(preferred_environment, list):
            raise TypeError("Preferred environment must be a list.")
        self._preferred_environment = preferred_environment

    # Getter & Setter for budget
    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, budget):
        if budget < 0:
            raise ValueError("Budget cannot be negative.")
        self._budget = budget


    def match_property_listing(self, property_listing):
        if self.budget >= property_listing.price_per_night:
            return True
        return False
    

    