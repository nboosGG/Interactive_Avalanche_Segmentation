
class attributes_converter():
    

    def avalanche_size_converter(s):
        s = str(s)
        if s == "1":
            return "ONE"
        elif s == "2":
            return "TWO"
        elif s == "3":
            return "THREE"
        elif s == "4":
            return "FOUR"
        elif s == "5":
            return "FIVE"
        else:
            return "UNKNOWN"


    def avalanche_type_converter(s):
        s = str(s)
        if s == "SLAB":
            return "SLAB"
        elif s == "GLIDE SNOW":
            return "FULL_DEPTH"
        elif s == "LOOSE SNOW":
            return "LOOSE_SNOW"
        else:
            return "UNKNOWN"
        
    def snow_moisture_converter(s):
        return str(s)
    
    def release_type_converter(s):
        s = str(s)
        if s == "NATURAL":
            return "NATURAL"
        elif s == "PERSON":
            return "PERSON"
        elif s == "EXPLOSIV":
            return "EXPLOSIVE"
        elif s == "SNOW GROOMER":
            return "SNOW_GROOMER"
        else:
            return "UNKNOWN"
        

    