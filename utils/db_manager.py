import sqlite3
import datetime
import pandas as pd
import json # JSON işlemleri için mutlaka ekle

def init_db():
    """Veritabanını ve gerekli tabloları başlatır."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    
    # 1. Kullanıcılar Tablosu
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE)''')
    
    # 2. Öğünler Tablosu - DETAILS SÜTUNU EKLENDİ
    c.execute('''CREATE TABLE IF NOT EXISTS meals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  date TEXT, 
                  meal_name TEXT, 
                  calories REAL, 
                  protein REAL, 
                  carb REAL, 
                  fat REAL,
                  details TEXT, -- Burası JSON string olarak tutulacak
                  FOREIGN KEY (user_id) REFERENCES users(user_id),
                  UNIQUE(user_id, date, meal_name) ON CONFLICT REPLACE)''')
    
    # 3. Profil Bilgileri Tablosu
    c.execute('''CREATE TABLE IF NOT EXISTS user_profile (
                    user_id INTEGER PRIMARY KEY,
                    gender TEXT,
                    age INTEGER,
                    height INTEGER,
                    weight REAL,
                    target_weight REAL,
                    activity TEXT,
                    daily_goal INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
    
    c.execute("INSERT OR IGNORE INTO users (user_id, username) VALUES (1, 'Admin')")
    conn.commit()
    conn.close()

# --- ÖĞÜN İŞLEMLERİ ---

def save_meal(user_id, meal_name, report):
    """Bir öğünü veritabanına kaydeder veya günceller."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    bugun = datetime.date.today().isoformat()
    
    # Report içindeki 'details' listesini JSON metnine çeviriyoruz
    details_json = json.dumps(report.get('details', []))
    
    c.execute("""INSERT OR REPLACE INTO meals 
                 (user_id, date, meal_name, calories, protein, carb, fat, details) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (user_id, bugun, meal_name, report['cal'], report['pro'], report['carb'], report['fat'], details_json))
    conn.commit()
    conn.close()

def get_today_meals_detailed(user_id):
    """Bugün girilen öğünlerin detaylarını (details dahil) getirir."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    bugun = datetime.date.today().isoformat()
    # Details sütununu SELECT sorgusuna ekledik
    c.execute("""SELECT meal_name, calories, protein, carb, fat, details 
                 FROM meals WHERE user_id=? AND date=?""", (user_id, bugun))
    meals = c.fetchall()
    conn.close()
    return meals

# Diğer fonksiyonların (get_daily_totals, delete_meal, get_weekly_data, vb.) 
# yapısında bir değişiklik yapmana gerek yok, onlar aynı kalabilir.


def get_daily_totals(user_id):
    """Bugünkü toplam kalori ve makro değerlerini döndürür."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    bugun = datetime.date.today().isoformat()
    c.execute("""SELECT SUM(calories), SUM(protein), SUM(carb), SUM(fat) 
                 FROM meals WHERE user_id=? AND date=?""", (user_id, bugun))
    totals = c.fetchone()
    conn.close()
    return [round(t, 1) if t else 0 for t in totals]

def delete_meal(user_id, meal_name):
    """Belirli bir öğünü siler."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    bugun = datetime.date.today().isoformat()
    c.execute("DELETE FROM meals WHERE user_id=? AND date=? AND meal_name=?", 
              (user_id, bugun, meal_name))
    conn.commit()
    conn.close()

def get_weekly_data(user_id):
    """Son 7 günlük verileri DataFrame olarak döndürür."""
    conn = sqlite3.connect('nutrition_data.db')
    query = """
    SELECT date, 
           SUM(calories) as total_cal, 
           SUM(protein) as total_pro, 
           SUM(carb) as total_carb, 
           SUM(fat) as total_fat
    FROM meals
    WHERE user_id = ? AND date >= date('now', '-6 days')
    GROUP BY date
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

# --- PROFİL İŞLEMLERİ (Yeni Fonksiyonlar) ---

def save_user_profile(user_id, gender, age, height, weight, target, activity, goal):
    """Profil bilgilerini kaydeder veya günceller."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO user_profile 
                 (user_id, gender, age, height, weight, target_weight, activity, daily_goal) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
              (user_id, gender, age, height, weight, target, activity, goal))
    conn.commit()
    conn.close()

def get_user_profile(user_id):
    """Kayıtlı profil bilgilerini getirir."""
    conn = sqlite3.connect('nutrition_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,))
    profile = c.fetchone()
    conn.close()
    return profile