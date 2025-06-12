import pandas as pd

class Personalization:
    def __init__(self):
        self.age_groups = {
            'young': (18, 30),
            'middle': (31, 45),
            'senior': (46, 100)
        }
        
    def personalize_by_age(self, recommendations, age):
        age_group = self._get_age_group(age)
        personalized = []
        
        for rec in recommendations:
            modified_rec = rec.copy()
            
            if age_group == 'young':
                modified_rec = self._adapt_for_young_adults(modified_rec)
            elif age_group == 'middle':
                modified_rec = self._adapt_for_middle_aged(modified_rec)
            else:
                modified_rec = self._adapt_for_seniors(modified_rec)
                
            personalized.append(modified_rec)
            
        return personalized
    
    def _get_age_group(self, age):
        for group, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group
        return 'middle'
    
    def _adapt_for_young_adults(self, recommendation):
        if recommendation['factor'] == 'screen_time':
            recommendation['recommendation'] += '. Consider using blue light filtering apps on devices.'
        elif recommendation['factor'] == 'stress_level':
            recommendation['recommendation'] += '. Try mobile apps for meditation and stress management.'
        return recommendation
    
    def _adapt_for_middle_aged(self, recommendation):
        if recommendation['factor'] == 'screen_time':
            recommendation['recommendation'] += '. Adjust workplace ergonomics and lighting.'
        elif recommendation['factor'] == 'sleep_quality':
            recommendation['recommendation'] += '. Consider establishing a consistent bedtime routine.'
        return recommendation
    
    def _adapt_for_seniors(self, recommendation):
        if recommendation['factor'] == 'blink_frequency':
            recommendation['recommendation'] += '. Regular eye exercises may help maintain muscle function.'
        elif recommendation['factor'] == 'humidity':
            recommendation['recommendation'] += '. Consider room humidifiers especially during heating season.'
        return recommendation
    
    def personalize_by_gender(self, recommendations, gender):
        personalized = []
        
        for rec in recommendations:
            modified_rec = rec.copy()
            
            if gender == 'F' or gender == 0:
                modified_rec = self._adapt_for_female(modified_rec)
            else:
                modified_rec = self._adapt_for_male(modified_rec)
                
            personalized.append(modified_rec)
            
        return personalized
    
    def _adapt_for_female(self, recommendation):
        if recommendation['factor'] == 'stress_level':
            recommendation['recommendation'] += '. Hormonal changes may affect dry eye; consult healthcare provider.'
        return recommendation
    
    def _adapt_for_male(self, recommendation):
        if recommendation['factor'] == 'physical_activity':
            recommendation['recommendation'] += '. Regular exercise can improve overall circulation including eye health.'
        return recommendation
    
    def personalize_by_lifestyle(self, recommendations, lifestyle_factors):
        personalized = []
        
        for rec in recommendations:
            modified_rec = rec.copy()
            
            if lifestyle_factors.get('work_type') == 'office':
                modified_rec = self._adapt_for_office_worker(modified_rec)
            elif lifestyle_factors.get('work_type') == 'outdoor':
                modified_rec = self._adapt_for_outdoor_worker(modified_rec)
                
            if lifestyle_factors.get('contact_lenses', False):
                modified_rec = self._adapt_for_contact_lens_user(modified_rec)
                
            personalized.append(modified_rec)
            
        return personalized
    
    def _adapt_for_office_worker(self, recommendation):
        if recommendation['factor'] == 'screen_time':
            recommendation['recommendation'] += '. Request ergonomic assessment of workstation.'
        elif recommendation['factor'] == 'humidity':
            recommendation['recommendation'] += '. Discuss office air quality with facility management.'
        return recommendation
    
    def _adapt_for_outdoor_worker(self, recommendation):
        if recommendation['category'] == 'Environment':
            recommendation['recommendation'] += '. Use protective eyewear in windy or dusty conditions.'
        return recommendation
    
    def _adapt_for_contact_lens_user(self, recommendation):
        if recommendation['factor'] == 'blink_frequency':
            recommendation['recommendation'] += '. Consider rewetting drops suitable for contact lenses.'
        elif recommendation['category'] == 'Medical':
            recommendation['recommendation'] += '. Inform eye care provider about contact lens use.'
        return recommendation
    
    def create_personalized_action_plan(self, recommendations, patient_profile):
        age = patient_profile.get('age', 30)
        gender = patient_profile.get('gender', 'M')
        lifestyle = patient_profile.get('lifestyle_factors', {})
        
        personalized_recs = self.personalize_by_age(recommendations, age)
        personalized_recs = self.personalize_by_gender(personalized_recs, gender)
        personalized_recs = self.personalize_by_lifestyle(personalized_recs, lifestyle)
        
        action_plan = {
            'immediate_actions': [rec for rec in personalized_recs if rec['priority'] in ['Critical', 'High']],
            'short_term_goals': [rec for rec in personalized_recs if rec['priority'] == 'Medium'],
            'long_term_maintenance': [rec for rec in personalized_recs if rec['priority'] == 'Low'],
            'patient_profile': patient_profile
        }
        
        return action_plan
    
    def generate_motivational_messages(self, patient_profile, progress=None):
        messages = []
        age_group = self._get_age_group(patient_profile.get('age', 30))
        
        if age_group == 'young':
            messages.append("Small changes now can prevent major eye problems later in life.")
        elif age_group == 'middle':
            messages.append("Taking care of your eyes supports your continued productivity and quality of life.")
        else:
            messages.append("Protecting your vision helps maintain independence and enjoyment of daily activities.")
            
        if progress and progress.get('improvements', []):
            messages.append("Great progress! Continue following your personalized plan.")
            
        return messages