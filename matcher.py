"""
Resume-Job Matcher Module
Implements scoring algorithms to match resumes against job descriptions
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeJobMatcher:
    def __init__(self):
        """Initialize the matcher with NLP models and similarity tools"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Weights for different scoring components
            self.weights = {
                'skills': 0.35,        # Skills matching is most important
                'experience': 0.25,    # Experience level matching
                'education': 0.15,     # Education requirements
                'semantic': 0.15,      # Semantic similarity of text
                'keywords': 0.10       # Keyword density matching
            }
            
        except Exception as e:
            logger.error(f"Error initializing ResumeJobMatcher: {e}")
            raise

    def calculate_skill_match_score(self, resume_skills: Dict[str, List[str]], 
                                  job_skills: Dict[str, List[str]]) -> Dict[str, any]:
        """Calculate skill matching score between resume and job"""
        
        # Flatten skills to lists
        resume_skills_flat = []
        job_skills_flat = []
        
        for skills in resume_skills.values():
            resume_skills_flat.extend([skill.lower() for skill in skills])
        
        for skills in job_skills.values():
            job_skills_flat.extend([skill.lower() for skill in skills])
        
        if not job_skills_flat:
            return {
                'score': 0.0,
                'matched_skills': [],
                'missing_skills': [],
                'match_percentage': 0.0,
                'details': 'No job skills found to match against'
            }
        
        # Find exact matches
        matched_skills = []
        for job_skill in job_skills_flat:
            if job_skill in resume_skills_flat:
                matched_skills.append(job_skill)
        
        # Find fuzzy matches for skills that didn't match exactly
        fuzzy_matched = []
        remaining_job_skills = [skill for skill in job_skills_flat if skill not in matched_skills]
        
        for job_skill in remaining_job_skills:
            best_match = None
            best_ratio = 0
            
            for resume_skill in resume_skills_flat:
                if resume_skill not in matched_skills:
                    ratio = fuzz.ratio(job_skill, resume_skill)
                    if ratio > best_ratio and ratio >= 80:  # 80% similarity threshold
                        best_ratio = ratio
                        best_match = resume_skill
            
            if best_match:
                fuzzy_matched.append((job_skill, best_match, best_ratio))
                matched_skills.append(job_skill)
        
        # Calculate scores
        total_job_skills = len(job_skills_flat)
        total_matched = len(matched_skills)
        match_percentage = (total_matched / total_job_skills) * 100 if total_job_skills > 0 else 0
        
        # Calculate weighted score (exact matches get full weight, fuzzy matches get partial)
        weighted_score = 0
        for skill in matched_skills:
            # Check if it was a fuzzy match
            fuzzy_match = next((fm for fm in fuzzy_matched if fm[0] == skill), None)
            if fuzzy_match:
                # Fuzzy matches get partial credit based on similarity ratio
                weighted_score += (fuzzy_match[2] / 100)
            else:
                # Exact matches get full credit
                weighted_score += 1.0
        
        final_score = min((weighted_score / total_job_skills) * 100, 100) if total_job_skills > 0 else 0
        
        missing_skills = [skill for skill in job_skills_flat if skill not in matched_skills]
        
        return {
            'score': final_score,
            'matched_skills': list(set(matched_skills)),
            'missing_skills': list(set(missing_skills)),
            'fuzzy_matches': fuzzy_matched,
            'match_percentage': match_percentage,
            'total_job_skills': total_job_skills,
            'total_matched': total_matched
        }

    def calculate_experience_match_score(self, resume_experience: Dict[str, any], 
                                       job_experience: Dict[str, any]) -> Dict[str, any]:
        """Calculate experience matching score"""
        
        resume_years = resume_experience.get('total_years', 0)
        job_min_years = job_experience.get('min_years', 0)
        job_max_years = job_experience.get('max_years', 0)
        job_level = job_experience.get('level', 'not_specified')
        
        # If no experience requirements, give neutral score
        if job_min_years == 0 and job_level == 'not_specified':
            return {
                'score': 70.0,  # Neutral score
                'details': 'No specific experience requirements found',
                'meets_requirement': True
            }
        
        # Calculate experience score
        if resume_years >= job_min_years:
            if job_max_years > 0:
                # If there's a range, being in range is optimal
                if resume_years <= job_max_years:
                    score = 100.0  # Perfect match within range
                else:
                    # Over-qualified but still good
                    excess_years = resume_years - job_max_years
                    penalty = min(excess_years * 5, 25)  # Max 25% penalty for overqualification
                    score = 100.0 - penalty
            else:
                # Only minimum specified, meeting it is good
                if resume_years == job_min_years:
                    score = 100.0
                else:
                    # More experience is better, but with diminishing returns
                    bonus = min((resume_years - job_min_years) * 10, 20)
                    score = 100.0 + bonus
                    score = min(score, 100.0)  # Cap at 100
        else:
            # Under-qualified
            shortage = job_min_years - resume_years
            penalty = shortage * 20  # 20% penalty per year short
            score = max(100.0 - penalty, 0.0)
        
        meets_requirement = resume_years >= job_min_years
        
        return {
            'score': score,
            'resume_years': resume_years,
            'required_min': job_min_years,
            'required_max': job_max_years,
            'meets_requirement': meets_requirement,
            'details': f"Resume: {resume_years} years, Required: {job_min_years}+ years"
        }

    def calculate_education_match_score(self, resume_education: Dict[str, List[str]], 
                                      job_education: Dict[str, List[str]]) -> Dict[str, any]:
        """Calculate education matching score"""
        
        resume_degrees = [deg.lower() for deg in resume_education.get('degrees', [])]
        job_required_degrees = [deg.lower() for deg in job_education.get('required_degrees', [])]
        
        # If no education requirements, give neutral score
        if not job_required_degrees:
            return {
                'score': 75.0,  # Neutral score
                'details': 'No specific education requirements found',
                'meets_requirement': True
            }
        
        # Check for degree matches
        degree_matches = []
        for job_degree in job_required_degrees:
            for resume_degree in resume_degrees:
                # Check for exact or fuzzy matches
                if (job_degree in resume_degree or resume_degree in job_degree or 
                    fuzz.ratio(job_degree, resume_degree) >= 70):
                    degree_matches.append((job_degree, resume_degree))
        
        # Calculate score based on matches
        if degree_matches:
            match_ratio = len(degree_matches) / len(job_required_degrees)
            score = match_ratio * 100
        else:
            # No degree matches, but check for general education level
            has_bachelor = any('bachelor' in deg or 'b.' in deg for deg in resume_degrees)
            has_master = any('master' in deg or 'm.' in deg for deg in resume_degrees)
            has_phd = any('phd' in deg or 'doctorate' in deg for deg in resume_degrees)
            
            requires_bachelor = any('bachelor' in deg or 'b.' in deg for deg in job_required_degrees)
            requires_master = any('master' in deg or 'm.' in deg for deg in job_required_degrees)
            requires_phd = any('phd' in deg or 'doctorate' in deg for deg in job_required_degrees)
            
            if requires_phd and has_phd:
                score = 100.0
            elif requires_master and (has_master or has_phd):
                score = 100.0
            elif requires_bachelor and (has_bachelor or has_master or has_phd):
                score = 100.0
            elif has_bachelor or has_master or has_phd:
                score = 60.0  # Has some degree but may not match requirement
            else:
                score = 30.0  # No matching degree level
        
        meets_requirement = len(degree_matches) > 0 or score >= 60.0
        
        return {
            'score': score,
            'degree_matches': degree_matches,
            'meets_requirement': meets_requirement,
            'details': f"Found {len(degree_matches)} degree matches out of {len(job_required_degrees)} requirements"
        }

    def calculate_semantic_similarity(self, resume_text: str, job_text: str) -> Dict[str, any]:
        """Calculate semantic similarity between resume and job description texts"""
        
        try:
            # Prepare texts
            texts = [resume_text, job_text]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0, 1] * 100  # Convert to percentage
            
            return {
                'score': similarity_score,
                'details': f"Semantic similarity: {similarity_score:.2f}%"
            }
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return {
                'score': 50.0,  # Default neutral score
                'details': f"Could not calculate semantic similarity: {e}"
            }

    def calculate_keyword_density_score(self, resume_text: str, job_text: str) -> Dict[str, any]:
        """Calculate keyword density matching score"""
        
        # Extract important keywords from job description
        job_doc = self.nlp(job_text.lower())
        job_keywords = []
        
        # Get nouns, proper nouns, and adjectives as keywords
        for token in job_doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                job_keywords.append(token.lemma_)
        
        # Count keyword occurrences in resume
        resume_text_lower = resume_text.lower()
        resume_keyword_count = 0
        
        for keyword in job_keywords:
            if keyword in resume_text_lower:
                resume_keyword_count += 1
        
        # Calculate density score
        if job_keywords:
            density_score = (resume_keyword_count / len(job_keywords)) * 100
        else:
            density_score = 50.0  # Neutral score if no keywords found
        
        return {
            'score': min(density_score, 100.0),
            'total_job_keywords': len(job_keywords),
            'matched_keywords': resume_keyword_count,
            'density_percentage': density_score,
            'details': f"Matched {resume_keyword_count} out of {len(job_keywords)} key terms"
        }

    def generate_recommendations(self, match_results: Dict[str, any]) -> List[str]:
        """Generate recommendations for improving resume-job match"""
        
        recommendations = []
        
        # Skills recommendations
        skills_result = match_results.get('skills', {})
        missing_skills = skills_result.get('missing_skills', [])
        if missing_skills:
            recommendations.append(f"Consider adding these skills: {', '.join(missing_skills[:5])}")
        
        # Experience recommendations
        exp_result = match_results.get('experience', {})
        if not exp_result.get('meets_requirement', True):
            req_years = exp_result.get('required_min', 0)
            recommendations.append(f"Gain more experience to meet the {req_years}+ years requirement")
        
        # Education recommendations
        edu_result = match_results.get('education', {})
        if not edu_result.get('meets_requirement', True):
            recommendations.append("Consider pursuing additional education/certifications")
        
        # General recommendations based on overall score
        overall_score = match_results.get('overall_score', 0)
        if overall_score < 70:
            recommendations.append("Consider tailoring your resume more closely to this job description")
            recommendations.append("Highlight relevant projects and accomplishments")
        
        return recommendations

    def match_resume_to_job(self, resume_data: Dict[str, any], 
                           job_data: Dict[str, any]) -> Dict[str, any]:
        """Main method to match resume against job description"""
        
        logger.info(f"Matching resume against job: {job_data.get('job_title', 'Unknown')}")
        
        # Calculate individual scores
        skills_match = self.calculate_skill_match_score(
            resume_data.get('skills', {}),
            job_data.get('required_skills', {})
        )
        
        experience_match = self.calculate_experience_match_score(
            resume_data.get('experience', {}),
            job_data.get('experience_requirements', {})
        )
        
        education_match = self.calculate_education_match_score(
            resume_data.get('education', {}),
            job_data.get('education_requirements', {})
        )
        
        semantic_similarity = self.calculate_semantic_similarity(
            resume_data.get('cleaned_text', ''),
            job_data.get('cleaned_text', '')
        )
        
        keyword_density = self.calculate_keyword_density_score(
            resume_data.get('cleaned_text', ''),
            job_data.get('cleaned_text', '')
        )
        
        # Calculate overall weighted score
        overall_score = (
            skills_match['score'] * self.weights['skills'] +
            experience_match['score'] * self.weights['experience'] +
            education_match['score'] * self.weights['education'] +
            semantic_similarity['score'] * self.weights['semantic'] +
            keyword_density['score'] * self.weights['keywords']
        )
        
        # Compile results
        match_results = {
            'overall_score': round(overall_score, 2),
            'skills': skills_match,
            'experience': experience_match,
            'education': education_match,
            'semantic_similarity': semantic_similarity,
            'keyword_density': keyword_density,
            'component_weights': self.weights,
            'job_title': job_data.get('job_title', 'Unknown'),
            'resume_file': resume_data.get('file_path', 'Unknown')
        }
        
        # Generate recommendations
        match_results['recommendations'] = self.generate_recommendations(match_results)
        
        # Determine match category
        if overall_score >= 85:
            match_results['match_category'] = 'Excellent Match'
        elif overall_score >= 70:
            match_results['match_category'] = 'Good Match'
        elif overall_score >= 55:
            match_results['match_category'] = 'Fair Match'
        else:
            match_results['match_category'] = 'Poor Match'
        
        logger.info(f"Resume matching completed. Overall score: {overall_score:.2f}%")
        
        return match_results

    def format_match_results(self, match_results: Dict[str, any]) -> str:
        """Format match results for display"""
        
        output = []
        output.append("=" * 50)
        output.append("RESUME ANALYSIS RESULTS")
        output.append("=" * 50)
        output.append("")
        
        # Overall results
        overall_score = match_results['overall_score']
        match_category = match_results['match_category']
        output.append(f"Overall Match Score: {overall_score}% ({match_category})")
        output.append(f"Job Title: {match_results['job_title']}")
        output.append("")
        
        # Skills analysis
        skills = match_results['skills']
        output.append("SKILL ANALYSIS:")
        output.append(f"✅ Found Skills: {', '.join(skills['matched_skills'][:10])}")
        if skills['missing_skills']:
            output.append(f"❌ Missing Skills: {', '.join(skills['missing_skills'][:5])}")
        output.append(f"📊 Skill Match: {skills['score']:.1f}%")
        output.append("")
        
        # Experience analysis
        experience = match_results['experience']
        exp_icon = "✅" if experience['meets_requirement'] else "❌"
        output.append("EXPERIENCE ANALYSIS:")
        output.append(f"{exp_icon} {experience['details']}")
        output.append(f"📊 Experience Match: {experience['score']:.1f}%")
        output.append("")
        
        # Education analysis
        education = match_results['education']
        edu_icon = "✅" if education['meets_requirement'] else "❌"
        output.append("EDUCATION ANALYSIS:")
        output.append(f"{edu_icon} {education['details']}")
        output.append(f"📊 Education Match: {education['score']:.1f}%")
        output.append("")
        
        # Additional metrics
        output.append("ADDITIONAL METRICS:")
        output.append(f"📝 Text Similarity: {match_results['semantic_similarity']['score']:.1f}%")
        output.append(f"🔍 Keyword Match: {match_results['keyword_density']['score']:.1f}%")
        output.append("")
        
        # Recommendations
        if match_results['recommendations']:
            output.append("RECOMMENDATIONS:")
            for i, rec in enumerate(match_results['recommendations'], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)


def main():
    """Test function for the matcher"""
    # This would typically use actual resume and job data
    print("Resume-Job Matcher initialized successfully!")
    print("Use this module by importing ResumeJobMatcher class")


if __name__ == "__main__":
    main()
