import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { useAppStore } from '../store/useAppStore';

const NotificationCenter: React.FC = () => {
  const { notifications, removeNotification } = useAppStore();

  if (notifications.length === 0) {
    return null;
  }

  const getNotificationStyle = (type: string) => {
    switch (type) {
      case 'success':
        return { bg: '#dcfce7', border: '#22c55e', text: '#16a34a', icon: '✓' };
      case 'error':
        return { bg: '#fee2e2', border: '#ef4444', text: '#dc2626', icon: '✕' };
      case 'warning':
        return { bg: '#fef3c7', border: '#f59e0b', text: '#d97706', icon: '⚠' };
      default:
        return { bg: '#dbeafe', border: '#3b82f6', text: '#2563eb', icon: 'ℹ' };
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {notifications.map((notification) => {
          const style = getNotificationStyle(notification.type);
          return (
            <View
              key={notification.id}
              style={[
                styles.notification,
                { backgroundColor: style.bg, borderLeftColor: style.border },
              ]}
            >
              <View style={styles.notificationContent}>
                <Text style={[styles.icon, { color: style.text }]}>{style.icon}</Text>
                <View style={styles.textContent}>
                  <Text style={[styles.title, { color: style.text }]}>
                    {notification.title}
                  </Text>
                  <Text style={styles.message}>{notification.message}</Text>
                </View>
                <TouchableOpacity
                  onPress={() => removeNotification(notification.id)}
                  style={styles.closeButton}
                >
                  <Text style={styles.closeText}>✕</Text>
                </TouchableOpacity>
              </View>
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 120,
    left: 10,
    right: 10,
    zIndex: 1000,
    maxHeight: 300,
  },
  scrollContent: {
    gap: 8,
  },
  notification: {
    borderRadius: 12,
    borderLeftWidth: 4,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  notificationContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  icon: {
    fontSize: 18,
    fontWeight: 'bold',
    marginRight: 10,
  },
  textContent: {
    flex: 1,
  },
  title: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
  },
  message: {
    fontSize: 13,
    color: '#4b5563',
  },
  closeButton: {
    padding: 4,
  },
  closeText: {
    fontSize: 16,
    color: '#9ca3af',
    fontWeight: 'bold',
  },
});

export default NotificationCenter;

