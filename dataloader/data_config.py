
data_config = {
	
	"semeval_2018_task7": {
		"rel2wgh": {
			"NONE": 0 , "COMPARE": 1, "MODEL-FEATURE": 0.5, "PART_WHOLE": 0.5,
			"RESULT": 1, "TOPIC": 5, "USAGE": 0.5,
		} , 

		"relations": ["COMPARE", "MODEL-FEATURE", "PART_WHOLE", "RESULT", "TOPIC", "USAGE", "NONE"] , 

		"sym_relations": ["COMPARE"] , 
	} , 

	"ace_2005": {
		"rel2wgh":{
			"PART-WHOLE": 1, "PHYS": 1, "GEN-AFF": 1, "ORG-AFF": 1, "ART": 1, 
			"PER-SOC": 1, "NO_RELATION": 0,
		} , 
		"relations": ["PART-WHOLE", "PHYS", "GEN-AFF", "ORG-AFF", "ART", "PER-SOC", "NO_RELATION"] , 

		"sym_relations": ["PER-SOC" , "PHYS"] , 
	}
}
